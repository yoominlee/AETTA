import torch

_CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
_CIFAR10_STDDEV = [0.2471, 0.2435, 0.2616]

_CIFAR100_MEAN = [0.5071, 0.4865, 0.4409]
_CIFAR100_STDDEV = [0.2673, 0.2564, 0.2762]

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STDDEV = [0.229, 0.224, 0.225]

_ES_MEAN = [0.4802, 0.4481, 0.3975]
_ES_STDDEV = [0.2302, 0.2265, 0.2262]

class NormalizeLayer(torch.nn.Module):
    """Standardize the channels of a batch of images by subtracting the dataset mean
      and dividing by the dataset standard deviation.
      """

    def __init__(self, means, sds):
        """
        :param means: the channel means
        :param sds: the channel standard deviations
        """
        super(NormalizeLayer, self).__init__()
        self.register_buffer(
            'mu', torch.tensor(means).view(-1, 1, 1))
        self.register_buffer(
            'sigma', torch.tensor(sds).view(-1, 1, 1))

    def forward(self, input: torch.tensor):
        return (input - self.mu) / self.sigma


def get_normalize_layer(dataset):
    """Return the dataset's normalization layer"""
    if dataset in ["cifar10", "cifar10outdist"]:
        return NormalizeLayer(_CIFAR10_MEAN, _CIFAR10_STDDEV)
    elif dataset in ["cifar100", "cifar100outdist"]:
        return NormalizeLayer(_CIFAR100_MEAN, _CIFAR100_STDDEV)
    elif dataset in ['imagenet', 'imagenetoutdist', 'imagenetA', 'imagenetR']:
        return NormalizeLayer(_IMAGENET_MEAN, _IMAGENET_STDDEV)
    elif dataset in ['es','esoutdist']:
        return NormalizeLayer(_ES_MEAN, _ES_STDDEV)
    else:
        raise NotImplementedError


def get_normalize_std(dataset):
    """Return the dataset's normalization layer"""
    if dataset in ["cifar10", "cifar10outdist"]:
        return _CIFAR10_STDDEV
    elif dataset in ["cifar100", "cifar100outdist"]:
        return _CIFAR100_MEAN, _CIFAR100_STDDEV
    elif dataset in ['imagenet', 'imagenetoutdist', 'imagenetA', 'imagenetR']:
        return _IMAGENET_MEAN, _IMAGENET_STDDEV
    elif dataset in ['es','esoutdist']:
        return _ES_MEAN, _ES_STDDEV
    else:
        raise NotImplementedError
