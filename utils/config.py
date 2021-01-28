import argparse


def argument_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--src", default="/SADNet-data/train_div2k.h5",
                        type=str, help="path for source file")
    parser.add_argument("--ckpt_dir", default="./ckpt/SADNET-synt/{}", type=str, help="directory path for checkpoints")
    parser.add_argument("--patch_size", default=128, type=int, help="the patch size")
    parser.add_argument("--batch_size", default=32, type=int, help="the batch size")
    parser.add_argument("--lr", default=1e-4, type=float, help="the learning rate")
    parser.add_argument("--num_epoch", default=200, type=int, help="number of epoch for training")
    parser.add_argument("--step_size", default=60, type=int, help="the epoch for learning rate decay")
    parser.add_argument("--gamma", default=0.5, type=float, help="gamma for scheduler")
    parser.add_argument("--num_channels", default=32, type=int, help="number of channels for the first layer")
    parser.add_argument("--offset_channels", default=32, type=int, help="number of offset channels")
    parser.add_argument("--sigma", default=30, type=int, help="sigma for noise level")
    parser.add_argument("--save_interval", default=5, type=int, help="save interval in type epoch")
    parser.add_argument("--wandb_interval", default=10, type=int, help="save interval for wandb in type step")
    parser.add_argument("--dataset", default="SIDD", type=str, help="dataset")
    parser.add_argument("--shuffle", action="store_true", help="shuffle the dataset or not")
    parser.add_argument("--real", action="store_true", help="real noise (true) or synthetic (false)")
    return parser
