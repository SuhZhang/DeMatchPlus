import argparse


def str2bool(v):
    return v.lower() in ("true", "1")

arg_lists = []
parser = argparse.ArgumentParser()

def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

# -----------------------------------------------------------------------------
# Network
net_arg = add_argument_group("Network")
net_arg.add_argument(
    "--layer_num", type=int, default=5, help=""
    "number of layers. Default: 5")
net_arg.add_argument(
    "--piece_num", type=int, default=48, help=""
    "number of pieces. Default: 48")
net_arg.add_argument(
    "--decrease_layer", type=list, default=[3, 4], help=""
    "list of cluster number.")
net_arg.add_argument(
    "--cluster_top_num", type=int, default=36, help=""
    "top number of matches in cluster. Default: 36")
net_arg.add_argument(
    "--neighbor_num", type=int, default=6, help=""
    "number of neighbors. Default: 6")
net_arg.add_argument(
    "--net_channels", type=int, default=128, help=""
    "number of channels in a layer. Default: 128")
net_arg.add_argument(
    "--head", type=int, default=4, help=""
    "number of head in attention. Default: 4")
net_arg.add_argument(
    "--layer_names", type=list, default=['cluster', 'cluster', 'context', 'context', 'context', 'context', 'decluster'], help=""
    "attention layer names in each filter layer.")
net_arg.add_argument(
    "--attention_mode", type=str, default='full', help=""
    "attention mode. Default: full")
net_arg.add_argument(
    "--use_fundamental", type=str2bool, default=False, help=""
    "train fundamental matrix estimation. Default: False")
net_arg.add_argument(
    "--use_ratio", type=int, default=0, help=""
    "use ratio test. 0: not use, 1: use before network, 2: use as side information. Default: 0")
net_arg.add_argument(
    "--use_mutual", type=int, default=0, help=""
    "use matual nearest neighbor check. 0: not use, 1: use before network, 2: use as side information. Default: 0")
net_arg.add_argument(
    "--ratio_test_th", type=float, default=0.8, help=""
    "ratio test threshold. Default: 0.8")

# -----------------------------------------------------------------------------
# Data
data_arg = add_argument_group("Data")
data_arg.add_argument(
    "--data_te", type=str, default='../data_dump/yfcc-sift-2000-test.hdf5', help=""
    "name of the unseen dataset for test")

# -----------------------------------------------------------------------------
# Filtering
filter_arg = add_argument_group("Test")
filter_arg.add_argument(
    "--model_file", type=str, default="../pretrained-model/yfcc100m", help=""
    "model file for test")
filter_arg.add_argument(
    "--gpu_id", type=str, default='0', help='id(s) for CUDA_VISIBLE_DEVICES')
filter_arg.add_argument(
    "--inlier_threshold", type=float, default=0., help=""
    "inlier threshold")

# -----------------------------------------------------------------------------
# test
test_arg = add_argument_group("Data")
test_arg.add_argument(
    "--use_ransac", type=str2bool, default=False, help=""
    "ransac as a robust estimator")
test_arg.add_argument(
    "--ransac_prob", type=float, default=0.99999, help=""
    "ransac prob value")
test_arg.add_argument(
    "--obj_geod_th", type=float, default=1e-4, help=""
    "theshold for the good geodesic distance")


def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed


def print_usage():
    parser.print_usage()

#
# config.py ends here
