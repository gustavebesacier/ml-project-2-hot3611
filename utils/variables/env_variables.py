import torch

# import utils.transforms
def set_device(force_cpu: bool=False) -> str:
    try:
        if not force_cpu:
            if torch.mps.is_available():
                return 'mps'
            elif torch.cuda.is_available():
                return 'cuda'
            else:
                return 'cpu'
    except:
        return 'cpu'
    else:
        return 'cpu'

def flatten(input):
    return input.flatten(0)
