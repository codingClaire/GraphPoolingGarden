import torch

def check_parameter(parameters):
    """set some default parameter not been set on json files
    """
    if 'device' not in parameters.keys():
        parameters['device'] = 0
    if 'num_workers' not in parameters.keys():
        parameters['num_workers'] = 0
    return parameters