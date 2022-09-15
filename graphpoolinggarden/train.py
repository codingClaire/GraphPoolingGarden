import torch
from torch_geometric.loader import DataLoader
import torch.optim as optim
from torch.utils.data import random_split

from tqdm import tqdm
import numpy as np
import json
import random
import os
import copy

from dataset.DatasetLoader import DatasetLoader
from utils.evaluate import metricsEvaluator
from utils.datasetaugment import get_dataset_info
from utils.argparser import get_argparser
from utils.parameter import check_parameter
from utils.file import save_csv

from nets.sequence_gnn_model import sequenceModel
from nets.hierarchical_gnn_model import hierarchicalModel

bcls_criterion = torch.nn.BCEWithLogitsLoss()
mcls_criterion = torch.nn.CrossEntropyLoss()
reg_criterion = torch.nn.MSELoss()


def train(model, device, loader, optimizer, task_type):
    model.train()

    for _, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            pred = model(batch)
            optimizer.zero_grad()
            ## ignore nan targets (unlabeled) when computing training loss.

            if "binary classification" in task_type:
                is_labeled = batch.y == batch.y
                if len(batch.y.shape) == 1:
                    batch.y = torch.unsqueeze(batch.y, 1)
                # for binary classification
                loss = bcls_criterion(
                    pred.to(torch.float32)[is_labeled],
                    batch.y.to(torch.float32)[is_labeled],
                )
            elif task_type == "multiple classification":
                loss = mcls_criterion(
                    pred.to(torch.float32),
                    batch.y.view(
                        -1,
                    )
                    # pred[is_labeled],
                    # batch.y[is_labeled],
                )
            else:
                loss = reg_criterion(
                    pred.to(torch.float32)[is_labeled],
                    batch.y.to(torch.float32)[is_labeled],
                )
            loss.backward()
            optimizer.step()


def eval(model, device, loader, evaluator, task_type):
    model.eval()
    y_true = []
    y_pred = []

    for _, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        with torch.no_grad():
            pred = model(batch)

        y_true.append(batch.y.view(-1, 1).detach().cpu())
        if task_type == "multiple classification":
            y_pred.append(torch.argmax(pred.detach(), dim=1).view(-1, 1).cpu())
        else:
            output = torch.sigmoid(pred.detach()).view(-1, 1).cpu()
            y_pred.append(
                torch.where(
                    output > 0.5, torch.ones_like(output), torch.zeros_like(output)
                )
            )

    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict)


def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


def main(net_parameters):
    net_parameters = check_parameter(net_parameters)
    device = (
        torch.device("cuda:" + str(net_parameters["device"]))
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    ### deal with dataset info
    dataset = DatasetLoader.load_dataset(net_parameters["dataset_name"])
    dataset_info = get_dataset_info(net_parameters["dataset_name"])
    net_parameters["in_dim"] = dataset_info["in_dim"]
    net_parameters["num_tasks"] = dataset_info["num_tasks"]
    # special operation for each dataset
    if net_parameters["dataset_name"] == "ogbg-molhiv":
        if net_parameters["feature"] == "full":
            pass
        elif net_parameters["feature"] == "simple":
            print("using simple feature")
            # only retain the top two node/edge features
            dataset.data.x = dataset.data.x[:, :2]
            dataset.data.edge_attr = dataset.data.edge_attr[:, :2]


    seed_everything(net_parameters["seed"])
    num_train = int(len(dataset) * 0.8)
    num_val = int(len(dataset) * 0.1)
    num_test = len(dataset) - (num_train + num_val)
    train_set, validation_set, test_set = random_split(
        dataset, [num_train, num_val, num_test]
    )

    train_loader = DataLoader(
        train_set,
        batch_size=net_parameters["batch_size"],
        shuffle=True,
        num_workers=net_parameters["num_workers"],
    )
    valid_loader = DataLoader(
        validation_set,
        batch_size=net_parameters["batch_size"],
        shuffle=False,
        num_workers=net_parameters["num_workers"],
    )
    test_loader = DataLoader(
        test_set,
        batch_size=net_parameters["batch_size"],
        shuffle=False,
        num_workers=net_parameters["num_workers"],
    )
    if(net_parameters["model"] == "hierarchical"):
        model = hierarchicalModel(net_parameters).to(device)
    else:
        # global model
        model = sequenceModel(net_parameters).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    valid_curve = []
    test_curve = []
    train_curve = []
    metrics_evaluator = metricsEvaluator(net_parameters["dataset_name"])
    for epoch in range(1, net_parameters["epochs"] + 1):
        print("=====Epoch {}".format(epoch))
        print("Training...")
        train(model, device, train_loader, optimizer, dataset_info["task_type"])

        print("Evaluating...")
        train_perf = eval(
            model, device, train_loader, metrics_evaluator, dataset_info["task_type"]
        )
        valid_perf = eval(
            model, device, valid_loader, metrics_evaluator, dataset_info["task_type"]
        )
        test_perf = eval(
            model, device, test_loader, metrics_evaluator, dataset_info["task_type"]
        )

        print({"Train": train_perf, "Validation": valid_perf, "Test": test_perf})

        train_curve.append(train_perf[dataset_info["metric"]])
        valid_curve.append(valid_perf[dataset_info["metric"]])
        test_curve.append(test_perf[dataset_info["metric"]])

    if "classification" in dataset_info["task_type"]:
        best_val_epoch = np.argmax(np.array(valid_curve))
        best_train = max(train_curve)
    else:
        best_val_epoch = np.argmin(np.array(valid_curve))
        best_train = min(train_curve)

    print("Finished training!")
    print("Best validation:")
    print("in score: {}".format(train_curve[best_val_epoch]))
    print("Validation score: {}".format(valid_curve[best_val_epoch]))
    print("Test score: {}".format(test_curve[best_val_epoch]))

    result = {
        "Val": valid_curve[best_val_epoch],
        "Test": test_curve[best_val_epoch],
        "Train": train_curve[best_val_epoch],
        "BestTrain": best_train,
    }
    return result


def repeat_experiment():
    parser = get_argparser()
    args = parser.parse_args()
    config = args.config 
    with open(config) as f:
        total_parameters = json.load(f)
    if(isinstance(total_parameters["seed"],list)):
        for seed in total_parameters["seed"]:
            if(isinstance(total_parameters["dataset_name"],list)):
                for dataset in total_parameters["dataset_name"]:
                    print("current seed: ", seed,
                          " current dataset: ", dataset)
                    net_parameters = copy.deepcopy(total_parameters)
                    net_parameters["seed"] = seed
                    net_parameters["dataset_name"] = dataset
                    result = main(net_parameters)
                    net_parameters.update(result)
                    save_csv(net_parameters,"result")



if __name__ == "__main__":
    repeat_experiment()
