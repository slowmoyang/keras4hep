from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os

def get_dataset_paths(min_pt):
    max_pt = int(min_pt * 1.1)

    hostname = os.environ["HOSTNAME"]
    if hostname == "cms05.sscc.uos.ac.kr":
        parent_dir = "/store/slowmoyang/QGJets"
    elif hostname == "gate2":
        parent_dir = "/home/scratch/slowmoyang/QGJets"
    elif hostname == "cms-gpu01.sdfarm.kr":
        parent_dir = "/cms/scratch/slowmoyang/QGJets"
    else:
        raise NotImplementedError

    format_str = os.path.join(
        parent_dir,
        "dijet_{min_pt}_{max_pt}/dijet_{min_pt}_{max_pt}_{{}}.root".format(
            min_pt=min_pt, max_pt=max_pt))

    paths = {key: format_str.format(key) for key in ["training", "validation", "test"]}
    paths["preprocessing"] = prep_path
    return paths
