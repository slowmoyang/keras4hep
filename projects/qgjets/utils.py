from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pandas as pd

from keras4hep.utils.misc import parse_str

def get_dataset_paths(min_pt, subset_type=None):
    max_pt = int(min_pt * 1.1)

    hostname = os.environ["HOSTNAME"]
    if hostname == "cms05.sscc.uos.ac.kr":
        parent_dir = "/store/slowmoyang/"
    elif hostname == "gate2":
        parent_dir = '/scratch/slowmoyang/'
    elif hostname == "cms-gpu01.sdfarm.kr":
        parent_dir = "/cms/scratch/slowmoyang/"
    else:
        raise NotImplementedError

    data_dir = os.path.join(parent_dir, "QGJets", "Data")

    format_str = os.path.join(
        data_dir,
        "dijet_{0}_{1}/dijet_{0}_{1}_{{}}.root".format(min_pt, max_pt))

    prep_path = os.path.join(
        parent_dir,
        "dijet_{0}_{1}/preprocessing_dijet_{0}_{1}_training.npz".format(min_pt, max_pt))

    if subset_type is None:
        paths = {key: format_str.format(key) for key in ["training", "validation", "test"]}
        paths["preprocessing"] = prep_path
        return paths
    elif subset_type in ["training", "validation", "test"]:
        return format_str.format(subset_type), prep_path
    else:
        raise NotImplementedError


def find_best_checkpoint(log_dir):
    if log_dir.endswith("/"):
        name = os.path.basename(log_dir.rstrip("/"))
    else:
        name = os.path.basename(log_dir)

    roc_curve_dir = os.path.join(log_dir, "roc_curve")
    roc_curve = [each for each in os.listdir(roc_curve_dir)]

    if len(roc_curve) == 0:
        return None

    roc_curve = [os.path.splitext(each)[0] for each in roc_curve]
    parsed_roc_curve = [parse_str(each) for each in roc_curve]
    best = max(parsed_roc_curve, key=lambda each: each["auc"])

    best["name"] = name

    ckpt_dir = os.path.join(log_dir, "checkpoint")
    if not os.path.exists(ckpt_dir):
        ckpt_dir = os.path.join(log_dir, "model_checkpoint")

    for each in os.listdir(ckpt_dir):
        name = os.path.splitext(each)[0]

        if parse_str(name, "epoch") == best["epoch"]:
            best_path = os.path.join(ckpt_dir, each)
            best_path = os.path.abspath(best_path)
            break

    best["path"] = best_path

    return best


def get_ranking(logs, write=True):
    log_dirs = [os.path.join(logs, each) for each in os.listdir(logs)]
    log_dirs = [each for each in log_dirs if os.path.isdir(each)]

    data = []
    for each in log_dirs:
        datum = find_best_checkpoint(each)
        if datum is None:
            continue
        data.append(datum)
    df = pd.DataFrame(data)
    df = df.sort_values(by="auc", ascending=False)

    if write:
        out_path = os.path.join(logs, "ranking.csv")
        df.to_csv(out_path)

    return df


