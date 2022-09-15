import os
import json

def get_dataset_info(dataset_name):
    with open(os.path.join(os.path.dirname(__file__), 'dataset_info.json'), encoding='utf-8') as f:
        dataset_info = json.load(f)
        if dataset_name not in dataset_info.keys():
            error_mssg = 'Invalid dataset name {}.\n'.format(dataset_name)
            error_mssg += 'Available datasets are as follows:\n'
            error_mssg += '\n'.join(dataset_info.keys())
            raise ValueError(error_mssg)
        #eval_metric = dataset_info[dataset_name]["metric"]
        #num_tasks = int(dataset_info[dataset_name]['num_tasks'])
        return dataset_info[dataset_name]