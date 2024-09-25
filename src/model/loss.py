import torch.nn as nn

def get_loss(loss_name):
    loss_dict = {
        "L1loss": nn.L1Loss(),
        "L2loss": nn.MSELoss(),
        "MSEloss": nn.MSELoss(),
        "BCEloss": nn.BCELoss(),
        "HuberLoss": nn.HuberLoss(),
        "RMSEloss": nn.MSELoss(),
        "SmoothL1loss": nn.SmoothL1Loss(),
    }
    try:
        return loss_dict[loss_name]
    except KeyError:
        raise NotImplementedError(f"Not Implemented Loss Function: {loss_name}")