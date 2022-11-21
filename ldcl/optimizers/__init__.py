import torch


def get_optimizer(name, model, lr, momentum, weight_decay, backbone=None, parameters=None):
    predictor_prefix = ('module.predictor', 'predictor')
    if not parameters:
        parameters = [{
            'name': 'base',
            'params': [param for name, param in model.named_parameters() if not name.startswith(predictor_prefix)],
            'lr': lr
        }, {
            'name': 'predictor',
            'params': [param for name, param in model.named_parameters() if name.startswith(predictor_prefix)],
            'lr': lr
        }]
    elif name == 'sgd':
        optimizer = torch.optim.SGD(parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif name == 'adam':
        optimizer = torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    else:
        raise NotImplementedError
    return optimizer