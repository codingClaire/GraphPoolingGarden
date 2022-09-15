import argparse


def add_gnn_model_arguments(parser: argparse.ArgumentParser) -> None:
    """add basic gnn model arguments

    Args:
        parser (argparse.ArgumentParser): _description_
    """
    # Training settings
    parser.add_argument(
        "--device", type=int, default=0, help="which gpu to use if any (default: 0)"
    )
    parser.add_argument(
        "--config", type= str ,help =" the dir of config file"
    )
    parser.add_argument(
        "--gnn",
        type=str,
        help="GNN gin, gin-virtual, or gcn, or gcn-virtual (default: gin-virtual)",
    )
    parser.add_argument(
        "--drop_ratio", type=float, default=0.5, help="dropout ratio (default: 0.5)"
    )
    parser.add_argument(
        "--num_layer",
        type=int,
        help="number of GNN message passing layers (default: 5)",
    )
    parser.add_argument(
        "--emb_dim",
        type=int,
        help="dimensionality of hidden units in GNNs (default: 300)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="input batch size for training (default: 32)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="number of epochs to train",
    )
    parser.add_argument(
        "--num_workers", type=int, default=0, help="number of workers (default: 0)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="dataset name (default: ogbg-molhiv)",
    )

    parser.add_argument(
        "--feature", type=str, default="full", help="full feature or simple feature"
    )
    parser.add_argument(
        "--filename", type=str, default="", help="filename to output result (default: )"
    )


def get_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    add_gnn_model_arguments(parser)
    return parser
