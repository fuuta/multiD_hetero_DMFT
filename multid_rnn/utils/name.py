from pathlib import Path
from ..dataclass.net_params import NetworkParameters
from ..dataclass.simulation_params import SimulationParameters
import json
from serde.json import to_json
from .logging_utils import get_logger

logger = get_logger()


def dict2string(ds):
    if isinstance(ds, list):
        name = ""
        for d in ds:
            name = name + "_".join(["{}{}".format(k, d[k]) for k in sorted(d.keys())])
    else:
        name = "_".join(["{}{}".format(k, ds[k]) for k in sorted(ds.keys())])
    return name


def determine_experiment_dirname(
    sim_params: SimulationParameters, net_params: NetworkParameters
):
    additinal_dir_prefix = sim_params.dir_prefix
    net_type = net_params.net_type
    hetero_info = net_params.hetero_info
    eI_params = net_params.external_input

    def pop_if_exists(d: dict, key: str):
        if key in d.keys():
            del d[key]

    dir_prefix = Path(additinal_dir_prefix)

    if net_type is None:
        dir_prefix = dir_prefix / net_params._net_type
    else:
        dir_prefix = (
            dir_prefix / net_type.__class__.__name__
        )  # ネットワークの種類をディレクトリ名に使用
    sim_params_dict = dict(json.loads(to_json(sim_params)))
    pop_if_exists(sim_params_dict, "dir_prefix")
    pop_if_exists(sim_params_dict, "is_compute_mle")
    pop_if_exists(sim_params_dict, "is_ret_dys")
    pop_if_exists(sim_params_dict, "n_block")
    trial_dir = Path("results") / dir_prefix / dict2string(sim_params_dict)
    net_params_dict = dict(json.loads(to_json(net_params)))
    hetero_info_dict = dict(json.loads(to_json(hetero_info)))
    pop_if_exists(net_params_dict, "hetero_info")
    pop_if_exists(net_params_dict, "net_type")

    if eI_params is not None:
        del net_params_dict["external_input"]
        eI_params_dict = dict(json.loads(to_json(eI_params)))
        trial_dir = (
            trial_dir
            / dict2string(hetero_info_dict)
            / dict2string(eI_params_dict)
            / dict2string(net_params_dict)
        )
    else:
        trial_dir = (
            trial_dir / dict2string(hetero_info_dict) / dict2string(net_params_dict)
        )
    return trial_dir
