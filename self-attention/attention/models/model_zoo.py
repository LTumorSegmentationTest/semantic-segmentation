# pylint: disable=wildcard-import, unused-wildcard-import

from .fcn import *
from .psp import *

__all__ = ['get_model']


def get_model(name, **kwargs):
    """Returns a pre-defined model by name

    Parameters
    ----------
    name : str
        Name of the model.
    pretrained : bool
        Whether to load the pretrained weights for model.
    root : str, default '~/.attention/models'
        Location for keeping the model parameters.

    Returns
    -------
    Module:
        The model.
    """
    models = {
        'fcn_resnet50_pcontext': get_fcn_resnet50_pcontext,
        'fcn_resnet50_ade': get_fcn_resnet50_ade,
        'psp_resnet50_ade': get_psp_resnet50_ade,
        }
    name = name.lower()
    if name not in models:
        raise ValueError('%s\n\t%s' % (str(e), '\n\t'.join(sorted(models.keys()))))
    net = models[name](**kwargs)
    return net
