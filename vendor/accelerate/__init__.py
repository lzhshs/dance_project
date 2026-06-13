import torch


class DistributedDataParallelKwargs:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class Accelerator:
    def __init__(self, *args, **kwargs):
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.is_main_process = True

    def prepare(self, *objects):
        if len(objects) == 1:
            return objects[0]
        return objects

    def wait_for_everyone(self):
        return None

    def backward(self, loss):
        loss.backward()
