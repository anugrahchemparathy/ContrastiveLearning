import torch

def get_device(cpu_only=False):
    """
        Returns device: either CPU, CUDA (if available), or
        Metal (Apple M1 chip).

        :param cpu_only: boolean whether only CPU should be
            returned.
        :return: torch.device object with best available object
    """
    if cpu_only:
        return torch.device("cpu")

    device = None
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        try:
            if torch.backends.mps.is_available():
                device = torch.device("mps")
        except:
            pass

    if device == None:
        device = torch.device("cpu")

    return device
