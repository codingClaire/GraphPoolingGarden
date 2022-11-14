import torch
from torch_geometric.loader import DataLoader
import torch.optim as optim
from torchvision import transforms

from tqdm import tqdm
import numpy as np
import pandas as pd
import json
import random
import os
import copy

from dataset.DatasetLoader import DatasetLoader
from utils.evaluate import metricsEvaluator
from utils.datasetaugment import get_dataset_info
from utils.datasetaugment import (
    get_vocab_mapping,
    augment_edge,
    encode_y_to_arr,
    decode_arr_to_seq,
)
from utils.argparser import get_argparser
from utils.parameter import check_parameter
from utils.file import save_csv_with_configname
from utils.train_util import load_epoch

from nets.sequence_gnn_model import sequenceModel
from nets.hierarchical_gnn_model import hierarchicalModel
from nets.graphUnet_model import graphUnetModel
from nets.diffpool_model import diffPoolModel

bcls_criterion = torch.nn.BCEWithLogitsLoss()
mcls_criterion = torch.nn.CrossEntropyLoss()
reg_criterion = torch.nn.MSELoss()


def train(model, device, loader, optimizer, task_type):
    model.train()

    for _, batch in enumerate(tqdm(loader, desc="Iteration",mininterval=60)):
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
            elif task_type == "subtoken prediction":
                # ogbg-code2
                loss = 0
                for i in range(len(pred)):
                    loss += mcls_criterion(pred[i].to(torch.float32), batch.y_arr[:, i])
                loss = loss / len(pred)
            else:
                loss = reg_criterion(
                    pred.to(torch.float32)[is_labeled],
                    batch.y.to(torch.float32)[is_labeled],
                )
            loss +=model.get_model_loss()
            loss.backward()
            optimizer.step()


def eval(model, device, loader, evaluator, task_type):
    model.eval()
    y_true = []
    y_pred = []

    for _, batch in enumerate(tqdm(loader, desc="Iteration",mininterval=60)):
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


def eval_for_code(model, device, loader, evaluator, arr_to_seq):
    model = model.to(device)
    model.eval()
    seq_ref_list = []
    seq_pred_list = []

    for _, batch in enumerate(tqdm(loader, desc="Iteration",mininterval=60)):
        batch = batch.to(device)
        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred_list = model(batch)

            mat = []
            for i in range(len(pred_list)):
                mat.append(torch.argmax(pred_list[i], dim=1).view(-1, 1))
            mat = torch.cat(mat, dim=1)

            seq_pred = [arr_to_seq(arr) for arr in mat]

            # PyG = 1.4.3
            # seq_ref = [batch.y[i][0] for i in range(len(batch.y))]

            # PyG >= 1.5.0
            seq_ref = [batch.y[i] for i in range(len(batch.y))]

            seq_ref_list.extend(seq_ref)
            seq_pred_list.extend(seq_pred)

    input_dict = {"seq_ref": seq_ref_list, "seq_pred": seq_pred_list}

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

    # net_parameters["device"] = torch.device("cpu")
    # device = net_parameters["device"]

    ### deal with dataset info
    dataset = DatasetLoader.load_dataset(net_parameters["dataset_name"])
    dataset_info = get_dataset_info(net_parameters["dataset_name"])
    net_parameters["in_dim"] = dataset_info["in_dim"]
    net_parameters["num_tasks"] = dataset_info["num_tasks"]
    ### set seed
    seed_everything(net_parameters["seed"])
    ### split dataset with the dataset_ratio
    # dataset_ration can be set on json file such as 0.1, 0.5 or so
    if "dataset_ratio" not in net_parameters.keys():
        net_parameters["dataset_ratio"] = 1
    total_num = int(len(dataset) * net_parameters["dataset_ratio"])
    perm = torch.randperm(total_num)
    num_train = int(total_num * 0.8)
    num_val = int(total_num * 0.1)
    num_test = total_num - (num_train + num_val)

    train_idx = perm[:num_train]
    valid_idx = perm[num_train : num_train + num_val]
    test_idx = perm[num_train + num_val :]

    assert len(train_idx) == num_train
    assert len(valid_idx) == num_val
    assert len(test_idx) == num_test
    ### whole dataset transform
    if net_parameters["dataset_name"] == "ogbg-molhiv":
        if net_parameters["feature"] == "full":
            pass
        elif net_parameters["feature"] == "simple":
            print("using simple feature")
            # only retain the top two node/edge features
            dataset.data.x = dataset.data.x[:, :2]
            dataset.data.edge_attr = dataset.data.edge_attr[:, :2]
    elif net_parameters["dataset_name"] == "ogbg-code2":
        ### building vocabulary for sequence predition. Only use training data.
        vocab2idx, idx2vocab = get_vocab_mapping(
            [dataset.data.y[i] for i in train_idx], net_parameters["num_vocab"]
        )

        nodetypes_mapping = pd.read_csv(
            os.path.join(dataset.root, "mapping", "typeidx2type.csv.gz")
        )
        nodeattributes_mapping = pd.read_csv(
            os.path.join(dataset.root, "mapping", "attridx2attr.csv.gz")
        )

        dataset.transform = transforms.Compose(
            [
                augment_edge,
                lambda data: encode_y_to_arr(
                    data, vocab2idx, net_parameters["max_seq_len"]
                ),
            ]
        )

        net_parameters["num_nodetypes"] = len(nodetypes_mapping["type"])
        net_parameters["num_nodeattributes"] = len(nodeattributes_mapping["attr"])
        net_parameters["max_depth"] = 20
        net_parameters["edge_dim"] = 2

    ########### split dataset #############
    # train_set, validation_set, test_set = random_split(
    #    dataset, [num_train, num_val, num_test]
    # )
    train_set, validation_set, test_set = (
        dataset[train_idx],
        dataset[valid_idx],
        dataset[test_idx],
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

    ####### initialize model (global/hieraraichal/graphunet) ######
    if net_parameters["model"] == "hierarchical":
        model = hierarchicalModel(net_parameters).to(device)
    elif net_parameters["model"] == "global":
        model = sequenceModel(net_parameters).to(device)
    elif net_parameters["model"] == "graphunet":
        model = graphUnetModel(net_parameters).to(device)
    elif net_parameters["model"] == "diffpoolmodel":
        model = diffPoolModel(net_parameters).to(device)
    ###### initialize
    optimizer_state_dict = None
    if net_parameters["load"] > 0:
        model_state_dict, optimizer_state_dict = load_epoch(net_parameters["model_path"], net_parameters["load"])
        model.load_state_dict(model_state_dict)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)

    valid_curve = []
    test_curve = []
    train_curve = []
    if net_parameters["dataset_name"] == "ogbg-code2":
        train_precision, test_precision, valid_precision = [],[],[]
        train_recall,test_recall,valid_recall = [],[],[]
    metrics_evaluator = metricsEvaluator(net_parameters["dataset_name"])
    for epoch in range(net_parameters["load"] + 1, net_parameters["epochs"] + 1):
        print("=====Epoch {}".format(epoch))
        print("Training...")
        train(model, device, train_loader, optimizer, dataset_info["task_type"])

        print("Evaluating...")
        if net_parameters["dataset_name"] == "ogbg-code2":
            train_perf = eval_for_code(model,device,train_loader,metrics_evaluator,
                arr_to_seq=lambda arr: decode_arr_to_seq(arr, idx2vocab),
            )
            valid_perf = eval_for_code(model,device,valid_loader,metrics_evaluator,
                arr_to_seq=lambda arr: decode_arr_to_seq(arr, idx2vocab),
            )
            test_perf = eval_for_code(model,device,test_loader,metrics_evaluator,
                arr_to_seq=lambda arr: decode_arr_to_seq(arr, idx2vocab),
            )
        else:
            train_perf = eval(model,device, train_loader, metrics_evaluator, dataset_info["task_type"])
            valid_perf = eval(model,device, valid_loader, metrics_evaluator, dataset_info["task_type"])
            test_perf = eval(model, device, test_loader, metrics_evaluator, dataset_info["task_type"])
    

        if net_parameters["dataset_name"] == "ogbg-code2":
            # add more metrics
            train_precision.append(train_perf['precision'])
            test_precision.append(test_perf['precision'])
            valid_precision.append(valid_perf['precision'])
            train_recall.append(train_perf['recall'])
            test_recall.append(test_perf['recall'])
            valid_recall.append(valid_perf['recall'])


        print({"Train": train_perf, "Validation": valid_perf, "Test": test_perf})
        train_curve.append(train_perf[dataset_info["metric"]])
        valid_curve.append(valid_perf[dataset_info["metric"]])
        test_curve.append(test_perf[dataset_info["metric"]])

        best_val_epoch = np.argmax(np.array(valid_curve))
        best_train = max(train_curve)
        print("current_best:")
        print("Train score: {}".format(train_curve[best_val_epoch]))
        print("Validation score: {}".format(valid_curve[best_val_epoch]))
        print("Test score: {}".format(test_curve[best_val_epoch]))
        if net_parameters["dataset_name"] == "ogbg-code2":
            print("Train Precision: {}, Recall: {}".format(train_precision[best_val_epoch],train_recall[best_val_epoch]))
            print("Valid Precision: {}, Recall: {}".format(valid_precision[best_val_epoch],valid_recall[best_val_epoch]))
            print("Test Precision: {}, Recall: {}".format(test_precision[best_val_epoch],test_recall[best_val_epoch]))
        
        if net_parameters["save_every_epoch"] and epoch % net_parameters["save_every_epoch"]  == 0:
            tqdm.write("saving to epoch.%04d.pth" % epoch)
            torch.save(
                (model.state_dict(), optimizer.state_dict()),
                os.path.join(net_parameters["model_path"], "epoch.%04d.pth" % epoch),
            )

    ############# end epoch ##################### 
    if "classification" in dataset_info["task_type"] or "prediction" in dataset_info["task_type"]:
        best_val_epoch = np.argmax(np.array(valid_curve))
        best_train = max(train_curve)
    else:
        # TODO ?
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
    config_name = config.split("/")[-1].split(".")[0]
    with open(config) as f:
        total_parameters = json.load(f)
    if isinstance(total_parameters["seed"], list):
        for seed in total_parameters["seed"]:
            if isinstance(total_parameters["dataset_name"], list):
                for dataset in total_parameters["dataset_name"]:
                    print("current seed: ", seed, " current dataset: ", dataset)
                    net_parameters = copy.deepcopy(total_parameters)
                    net_parameters["seed"] = seed
                    net_parameters["dataset_name"] = dataset
                    result = main(net_parameters)
                    net_parameters.update(result)
                    save_csv_with_configname(net_parameters,"result",config_name)


if __name__ == "__main__":
    repeat_experiment()
