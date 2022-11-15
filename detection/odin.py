import torch
from tqdm import tqdm

def get_odin_score(model, loader, criterion, noise_magnitude=0.0014, temperature=1000, device=None):
    model.eval()
    score = []
    for data in tqdm(loader):
        inputs = data[0].to(device)
        inputs.requires_grad_()
        outputs = model(inputs)
        outputs = outputs / temperature

        max_index = torch.argmax(outputs, dim=1).detach()
        loss = criterion(outputs, max_index)
        loss.backward()

        gradient = torch.ge(inputs.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2
        # Normalizing the gradient to the same space of image
        # print(gradient.shape)
        gradient[:, 0, :, :] = (gradient[:, 0, :, :] )/(63.0/255.0)
        gradient[:, 1, :, :] = (gradient[:, 1, :, :] )/(62.1/255.0)
        gradient[:, 2, :, :] = (gradient[:, 2, :, :])/(66.7/255.0)

        new_inputs = inputs.data - noise_magnitude * gradient
        with torch.no_grad():
            outputs = model(new_inputs)
        outputs = outputs / temperature
        probs = outputs.softmax(dim=1)
        score.append(probs.max(dim=1)[0])
    return torch.cat(score)