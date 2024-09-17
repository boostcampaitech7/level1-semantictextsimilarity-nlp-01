import torch.optim as optim

def get_optimizer(optimizer_name):
    optimizer_dict = {
        "Adam": optim.Adam,
        "AdamW": optim.AdamW,
        "SGD": optim.SGD,
    }
    try:
        return optimizer_dict[optimizer_name]
    except:
        raise NotImplementedError(f"Not Implemented Optimizer: {optimizer_name}")