import torch
import os

def check_parameter(parameters):
    """set some default parameter not been set on json files
    """
    if 'device' not in parameters.keys():
        parameters['device'] = 0
    if 'num_workers' not in parameters.keys():
        parameters['num_workers'] = 0
    if not os.path.exists(parameters["model_path"]):
        os.makedirs(parameters["model_path"])
    return parameters