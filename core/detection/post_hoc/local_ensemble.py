import torch
import torch.nn.functional as F
from .base_detector import BaseDetector
from functorch import make_functional_with_buffers, make_functional, vmap, vjp, grad, jvp,  grad_and_value
from scipy.linalg import eigh_tridiagonal
import optree
# from hessian_eigenthings import compute_hessian_eigenthings

from tqdm import tqdm
from time import time
import logging

logger = logging.getLogger(__name__)

def tree_flatten(v):
    def f(v):
        leaves, _ = optree.tree_flatten(v)
        return torch.cat([x.view(-1) for x in leaves])
    out, pullback = vjp(f, v)
    return out, lambda x: pullback(x)[0]

class LocalEnsembleDetector(BaseDetector):
    def __init__(self, net, num_eigen_to_compute, m, **kwargs) -> None:
        super().__init__("local_ensemble", net, False)
        self.m = m
        self.num_eigen_to_compute = num_eigen_to_compute

        f_fc, _params, _buffers = make_functional_with_buffers(self.net, disable_autograd_tracking=True)
        self._flatten_params, self._unflatten = tree_flatten(_params)
        print(self._flatten_params.shape)
        
        self.net.eval()
        for param in self.net.parameters():
            param.requires_grad = False

        def forward(flatten_params, x):
            # out = self.net.conv1(x)
            # out = self.net.block1(out)
            # out = self.net.block2(out)
            # out = f_fc(self._unflatten(flatten_params), _buffers, out)
            # out = self.net.relu(self.net.bn1(out))
            # out = F.avg_pool2d(out, 8)
            # out = out.view(-1, self.net.nChannels)
            # return self.net.fc(out)
            return f_fc(self._unflatten(flatten_params), _buffers, x)[0]
            # out = self.net.pre_linear(x)
            # out = f_fc(self._unflatten(flatten_params), out)
            # return out

        self._forward = forward


    def setup(self, train_loader, **kwargs):
        t0 = time()
        device = self._flatten_params.device
        def loss_fn(flatten_params, data, target):
            output = self._forward(flatten_params, data)
            return F.mse_loss(output, target, reduction='none').mean()
        test_batch = next(iter(train_loader))
        test_batch = (test_batch[0].to(device), test_batch[1].to(device))
        print("Loss:", loss_fn(self._flatten_params, *test_batch).item())
        print("Test grad:",  _grad_batch(loss_fn, test_batch, self._flatten_params).size())
        print("Test hvb:",  _hvp_batch(loss_fn, test_batch, self._flatten_params, torch.ones_like(self._flatten_params)).size())

        # self.H_evals, self.H_evecs = compute_hessian_eigenthings(self.net, train_loader, F.cross_entropy, self.m, mode='lanczos')
        # print(self.H_evecs.shape)
        self.H_evals, self.H_evecs = build_ensemble_space(loss_fn, self._flatten_params, train_loader, self.num_eigen_to_compute)
        print(self.H_evecs.shape)

        def pred(flatten_params, x):
            x = x.unsqueeze(0)
            output = self._forward(flatten_params, x)
            output = output.squeeze()
            # score = output.mean() - torch.logsumexp(output, dim=0)
            score = output
            # score = F.softmax(output, dim=0).max()
            return score, output
        self._pred_grads_fn = vmap(grad_and_value(pred, has_aux=True), in_dims=(None, 0))
        
        logger.info(f"Training set singular values largest: {self.H_evals[:5]}")
        logger.info(f"Finish build ensemble space in {time() -t0 :.2f} sec")
    
    def score_batch(self, data: torch.Tensor):
        data = data.to(self._flatten_params.device)
        loss_grads, output = self._pred_grads_fn(self._flatten_params, data)
        output = output[1]
        proj_grads = torch.matmul(
                        torch.matmul(loss_grads, self.H_evecs[:,:self.m]), 
                        self.H_evecs[:,:self.m].T
                    )
        small_grads = loss_grads - proj_grads
        small_grad_norms = torch.linalg.norm(small_grads, dim=1, keepdims=True)

        # return small_grad_norms, torch.argmax(output, dim=1)
        return small_grad_norms, output

def _grad_batch(f, batch, params) -> torch.Tensor:
    """
    Compute gradient w.r.t loss over all parameters and vectorize
    """
    max_possible_gpu_samples = 255
    all_inputs, all_targets = batch
    num_chunks = max(1, len(all_inputs) // max_possible_gpu_samples)
    grad_vec = None

    # This will do the "gradient chunking trick" to create micro-batches
    # when the batch size is larger than what will fit in memory.
    # WARNING: this may interact poorly with batch normalization.
    input_microbatches = all_inputs.chunk(num_chunks)
    target_microbatches = all_targets.chunk(num_chunks)
    for input, target in zip(input_microbatches, target_microbatches):
        grads = grad(f, argnums=0)(params, input, target)
        if grad_vec is not None:
            grad_vec += grads
        else:
            grad_vec = grads
    grad_vec /= num_chunks
    return grad_vec

def _hvp_batch(f, batch, flatten_params, flatten_tangents):
    return jvp(lambda params: _grad_batch(f, batch, params) , (flatten_params, ), (flatten_tangents, ))[1]

def hvp(f, dataloader, flatten_params, flatten_tangents):
    n = len(dataloader)
    device = flatten_params.device
    hessian_vec_prod = None
    _hvp_batch_vmap = vmap(_hvp_batch, in_dims=(None, None, None, 1), out_dims=1)
    for batch in dataloader:
        batch = (batch[0].to(device), batch[1].to(device))
        if hessian_vec_prod is not None:
            hessian_vec_prod += _hvp_batch_vmap(f, batch, flatten_params, flatten_tangents)
        else:
            hessian_vec_prod = _hvp_batch_vmap(f, batch, flatten_params, flatten_tangents)
    hessian_vec_prod = hessian_vec_prod / n
    return hessian_vec_prod


def build_ensemble_space(loss_fn, params: torch.Tensor, dataloader, m, eps=1e-8):
    '''
    params: a vector
    '''
    device = params.device
    dim = params.shape[0]
    # implicit_hvp = vmap(lambda v: hvp(loss_fn, dataloader, params, v), 
    #                     in_dims=1, 
    #                     out_dims=1,
    #                     randomness='same')
    implicit_hvp = lambda v: hvp(loss_fn, dataloader, params, v)
    Q, Beta, Alpha = lanczos_iteration(implicit_hvp, 
                                       (dim, 1), m,
                                        eps=eps, 
                                        two_reorth=True, 
                                        dtype=torch.float32, 
                                        device=device)
    Q_lan = Q[:, 1:-1]
    # find these eigenvalues + vectors from the tridiagonalized matrix
    T_evals, T_evecs = eig_from_tridiagonal(Alpha[1:].cpu().numpy(), 
                                            Beta[1:-1].cpu().numpy())
    T_evals = torch.tensor(T_evals, device=device)
    T_evecs = torch.tensor(T_evecs, device=device)
    T_evals, T_evecs = sort_eigens_by_absval(T_evals, T_evecs)

    # if algorithm terminated early, remove smallest estimate
    if Beta[-1] < eps:
        T_evals = T_evals[:-1]
        T_evecs = T_evecs[:,:-1]

    # and derive the eigenvectors of A from the eigenvectors of T
    A_evals = T_evals
    A_evecs = torch.matmul(Q_lan, T_evecs)
    A_evals, A_evecs = sort_eigens_by_absval(A_evals, A_evecs)

    return A_evals, A_evecs

def lanczos_iteration(A_fn, shape, num_iters, eps=1e-8, dtype=torch.float32, two_reorth=False, device='cpu'):
    Q = [torch.zeros(shape, dtype=dtype, device=device)] # will collect Lanczos vectors here
    Beta = [1] # collect off-diagonal terms
    Alpha = [torch.FloatTensor([[0.]])] # collect diagonal terms

    # initialize
    q = torch.rand(size=shape,dtype=dtype, device=device)
    q /= torch.linalg.norm(q)
    r = q
    Q.append(q)
    Q_k_range = Q[0]
    for k in range(1, num_iters + 1):
        z = A_fn(Q[k])
        alpha = torch.matmul(Q[k].T, z)
        Alpha.append(alpha)

        Q_k_range = torch.cat([Q_k_range, Q[k]], dim=1)
        z_orth = torch.sum(torch.matmul(z.T, Q_k_range) * Q_k_range, axis=1, keepdims=True)
        z = z - z_orth

        if two_reorth:
            # re-orthogonalizing twice improves stability
            z_orth = torch.sum(torch.matmul(z.T, Q_k_range) * Q_k_range, axis=1, keepdims=True)
            z = z - z_orth
        r = z
        beta = torch.linalg.norm(r)
        Beta.append(beta)
        q = r / beta
        Q.append(q)
        if (k - 1) % 100 == 0:
            print('{:d} Lanczos iterations complete.'.format(k))
        if beta < eps:
            break

    Q = torch.cat(Q, dim=1)
    Alpha = torch.tensor(Alpha)
    Beta = torch.tensor(Beta)   
    return Q, Beta, Alpha

def eig_from_tridiagonal(alpha, beta):
    # Alpha_array = alpha.squeeze()
    # # Beta_array = beta.squeeze()
    # print(Alpha_array.shape, Beta_array.shape)
    T_evals, T_evecs = eigh_tridiagonal(alpha, beta)
    return T_evals, T_evecs

def sort_eigens_by_absval(evals, evecs):
    _, indices =  torch.sort(evals.abs(), descending=True)
    evals = evals[indices]
    evecs = evecs[:,indices]
    return evals, evecs