import torch
from torch import nn
from torch.autograd import Variable
from .base_detector import BaseDetector


class ODINDetector(BaseDetector):
    def __init__(self, net, T, noise_magnitude, *args, **kwargs) -> None:
        super().__init__('odin', net)
        self.T = T
        self.noise_magnitude=noise_magnitude
        self.require_grad = True
        self.net.eval()
    
    def score_batch(self, data):
        data = data.requires_grad_()
        output = self.net(data)
        preds = torch.argmax(output.detach(), dim=1)
        odin_score = ODIN(data, output, self.net, self.T, self.noise_magnitude)
        
        # minus since lower score is more ID
        return {
            "scores": -odin_score.detach(), 
            "preds": preds
        }
    
    def __str__(self) -> str:
        return f"{self.name}_T={self.T}_noise={self.noise_magnitude}"


def ODIN(inputs, outputs, model, temper, noiseMagnitude1):
    # Calculating the perturbation we need to add, that is,
    # the sign of gradient of cross entropy loss w.r.t. input
    criterion = nn.CrossEntropyLoss()

    # Using temperature scaling
    outputs = outputs / temper
    maxIndexTemp = torch.argmax(outputs.detach(), dim=1)

    loss = criterion(outputs, maxIndexTemp)
    loss.backward()

    # Normalizing the gradient to binary in {0, 1}
    gradient =  torch.ge(inputs.grad.data, 0)
    gradient = (gradient.float() - 0.5) * 2
    
    # Normalizing the gradient to the same space of image
    gradient[:,0] = (gradient[:,0] )/(63.0/255.0)
    gradient[:,1] = (gradient[:,1] )/(62.1/255.0)
    gradient[:,2] = (gradient[:,2] )/(66.7/255.0)

    # Adding small perturbations to images
    tempInputs = torch.add(inputs.data, gradient, alpha=-noiseMagnitude1)
    with torch.no_grad():
        outputs = model(tempInputs)
    outputs = outputs / temper
    # Calculating the confidence after adding perturbations
    probs = outputs.softmax(dim=1)
    nnOutputs = torch.max(probs, dim=1).values
    return nnOutputs