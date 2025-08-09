import json
import pickle
import re
from pathlib import Path
from typing import Literal

import jax
import numpy as np
import pandas as pd
import tqdm
from line_profiler import profile
from serde.json import to_json

from multid_rnn.utils.logging_utils import get_logger

logger = get_logger()

# if os.environ['JAX_CPU']=='true':
#     print("run JAX with CPU")
#     jax.config.update('jax_platform_name', 'cpu')
# else:
#     print("run JAX with GPU")
#     os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
#     from jax.config import config; config.update('jax_enable_x64', True)

from collections.abc import MutableMapping

from multid_rnn.dataclass.net_params import NetworkParameters


def _flatten_dict_gen(d, parent_key, sep):
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            yield from flatten_dict(v, new_key, sep=sep).items()
        else:
            yield new_key, v


def flatten_dict(d: MutableMapping, parent_key: str = "", sep: str = "_"):
    """
    入れ子になってる辞書をkey+valを新しくkeyとした入れ子じゃない辞書に展開する関数

    Args:
        d (MutableMapping): _description_
        parent_key (str, optional): _description_. Defaults to ''.
        sep (str, optional): _description_. Defaults to '_'.

    Returns:
        _type_: _description_
    """
    # https://www.freecodecamp.org/news/how-to-flatten-a-dictionary-in-python-in-4-different-ways/
    return dict(_flatten_dict_gen(d, parent_key, sep))


def _convert_dict_element(x):
    if isinstance(x, jax.Array) or isinstance(x, np.ndarray):
        if x.size == 1:
            return x.item()
    elif isinstance(x, dict):
        return x
    elif isinstance(x, list):
        return np.asarray(x)
    else:
        return x


def _dataclass_to_dict(obj: object) -> dict:
    return dict(json.loads(to_json(obj)))


def npz_to_df(npz_object: np.lib.npyio.NpzFile) -> pd.DataFrame:
    return pd.DataFrame.from_dict(
        dict(zip(npz_object.files, [npz_object[key] for key in npz_object.files]))
    )


SAVED_DATA_TYPES = Literal[
    "SxSphix", "MC", "heterovars", "idoled_xtrace", "relaxation_time"
]


def load_trial(
    net_path: Path,
    trialid: str,
    targetfile: SAVED_DATA_TYPES,
):
    if targetfile == "SxSphix":
        # saved-*.npz の読み込み (freqRangeを列としたSxなどの系列を含むDataframe)
        if (net_path / f"saved-{trialid}.npz").exists():
            trialsaved_p = net_path / f"saved-{trialid}.npz"
            trialsaved = np.load(trialsaved_p, allow_pickle=True)
            trialsaved.files = [
                k for k in trialsaved.files if k not in ["autocorrs", "autocorrs_lags"]
            ]  # autocorrsはshapeが違うので一旦排除
        elif (net_path / f"SxSphix-{trialid}.npz").exists():
            trialsaved_p = net_path / f"SxSphix-{trialid}.npz"
            trialsaved = np.load(trialsaved_p, allow_pickle=True)
        else:
            raise ValueError()
        df = npz_to_df(trialsaved)
        df["trialseed"] = trialid
        return df
    elif targetfile == "MC":
        MC_npz_list = (net_path / "MC").glob(f"**/*-{trialid}.npz")
        dfs = []
        for mc_npz in MC_npz_list:
            trialsaved = np.load(mc_npz, allow_pickle=True)
            tag = mc_npz.stem.split("-")[:-2]  # 末尾のtrialidを除いた部分がtag
            df = npz_to_df(trialsaved)
            df["trialseed"] = trialid
            df["tag"] = "-".join(tag)
            dfs.append(df)
        dfs = pd.concat(dfs, ignore_index=True)
        return dfs
    elif targetfile == "heterovars":
        trialsaved_p = net_path / f"hetero_var_vals-{trialid}.npz"
        if trialsaved_p.exists():
            trialsaved = np.load(trialsaved_p, allow_pickle=True)
            df = npz_to_df(trialsaved)
            df["trialseed"] = trialid
            return df
    elif targetfile == "idoled_xtrace":
        trialsaved_p = net_path / f"idoled_xtrace-{trialid}.npz"
        if trialsaved_p.exists():
            trialsaved = np.load(trialsaved_p, allow_pickle=True)
            df = npz_to_df(trialsaved)
            df["trialseed"] = trialid
            return df
    elif targetfile == "relaxation_time":
        trialsaved_p = net_path / f"relaxation_time-{trialid}.npz"
        if trialsaved_p.exists():
            trialsaved = np.load(trialsaved_p, allow_pickle=True)
            df = npz_to_df(trialsaved)
            df["trialseed"] = trialid
            return df
    else:
        raise ValueError(f"Unknown targetfile: {targetfile}")


def is_filtered_params(filter: dict | None, params: dict, is_pass: bool) -> bool:
    if filter is None:
        return True

    for key in filter.keys():
        if is_pass:
            if params.get(key) != filter.get(key):
                return False
        else:
            filter_value = filter.get(key)
            if isinstance(filter_value, list):
                if params.get(key) in filter_value:
                    return False
            else:
                if params.get(key) == filter_value:
                    return False
    return True


# def safe_get_nested(data, keys):
#     """安全に入れ子の値を取得"""
#     for key in keys:
#         if isinstance(data, dict) and key in data:
#             data = data[key]
#         else:
#             return None
#     return data

# def is_filtered_neted_dict(filter, _dict):
#     if filter is None:
#         return True

#     for key in filter.keys():
#         dict_val = safe_get_nested(_dict, key)
#         if dict_val is not None:
#             if dict_val == filter[key]:
#                 continue
#             else:
#                 return False
#     return True


def _check_coeff(net_params_flat_dict: dict, coeff_filter: dict | None = None):
    if coeff_filter is None:
        return True
    coefficient = np.asarray(net_params_flat_dict["coefficient"])
    return (
        coefficient[coeff_filter["index"][0], coeff_filter["index"][1]]
        == coeff_filter["value"]
    )


def extract_last_two_ints(text):
    """文字列から後ろの2つの整数を取得する"""
    # 数字のパターンを全て抽出
    numbers = re.findall(r"\d+", text)

    if len(numbers) == 2:
        # 後ろの2つを整数として返す
        return f"{numbers[0]}-{numbers[1]}"
    else:
        raise ValueError(
            f"Expected two integers in the string, but found {len(numbers)}: {text}"
        )


def load_net(
    sim_path: Path,
    recursive_df: bool = True,
    filter_pass: dict | None = None,
    filter_reject: dict | None = None,
    coeff_filter: dict | None = None,
    list_read_dataname: list[SAVED_DATA_TYPES] = ["SxSphix", "heterovars"],
):
    """
    あるパス以下のすべての実験結果を読み込み，
    列が1つのネットワークの結果となるDataframe形式で返す
    結果がarrayやmatrixになる場合(SxやMC)はDataframe内の値がDataframeやndarrayになる
    """
    net_plist = [x.parent for x in sim_path.glob("**/net_params.pkl")]
    logger.info(f"{len(net_plist)} types of network in dir")
    df_net_list = []
    for net_p in tqdm.tqdm(
        net_plist, desc="load data from 1 network parameters", leave=False
    ):
        # ネットワークパラメタの読み込み
        net_p = Path(net_p)
        net_params_path = net_p / "net_params.pkl"
        with open(net_params_path, "rb") as f:
            net_params_dataclass: NetworkParameters = pickle.load(f)
        net_params_dict: dict = _dataclass_to_dict(
            net_params_dataclass
        )  # NetworkParameters dataclass をdictに変換

        if (net_p / "g_c.txt").exists():
            g_c = np.loadtxt(net_p / "g_c.txt")  # g_c.txtがあれば読み込む
            net_params_dict["critical_coupling_strength"] = g_c  # g_cを追加
        else:
            net_params_dict["critical_coupling_strength"] = (
                None  # g_c.txtがなければNone
            )

        net_params_dict = {
            k: _convert_dict_element(v) for k, v in net_params_dict.items()
        }  # sizeが1のjax.Arrayが紛れ込んでるので，それをscalarに変換
        # if not is_filtered_params(filter, net_params_dict):
        #     logger.info("skip {}".format(net_p))
        #     continue
        net_params_flat_dict = flatten_dict(net_params_dict)  # 入れ子Dictをflattenする
        if (
            not is_filtered_params(filter_pass, net_params_flat_dict, is_pass=True)
            or not is_filtered_params(
                filter_reject, net_params_flat_dict, is_pass=False
            )
            or not _check_coeff(net_params_flat_dict, coeff_filter)
        ):
            # logger.info("skip {}".format(net_p))
            continue
        df_net_params = pd.DataFrame([net_params_flat_dict])
        df_net_params["pkl_path"] = (
            net_params_path  # pkl_pathを列に追加 (後でg_cなどを計算できるように)
        )

        ############## 対象ネットワークパラメタにおける複数トライアルの読み込み ##############
        dict_df = {}
        trialid_list = [
            extract_last_two_ints(p.stem) for p in list(net_p.glob("*.done"))
        ]
        if len(trialid_list) == 0:
            continue

        if "SxSphix" in list_read_dataname:
            df_SxSphi_list = [
                load_trial(net_p, trialid, targetfile="SxSphix")
                for trialid in trialid_list
            ]
        else:
            df_SxSphi_list = [None] * len(trialid_list)
        dict_df["df_SxSphi"] = df_SxSphi_list

        if "heterovars" in list_read_dataname:
            df_hetero_list = [
                load_trial(net_p, trialid, targetfile="heterovars")
                for trialid in trialid_list
            ]
        else:
            df_hetero_list = [None] * len(trialid_list)
        dict_df["df_hetero"] = df_hetero_list

        if "idoled_xtrace" in list_read_dataname:
            df_xtrace_list = [
                load_trial(net_p, trialid, targetfile="idoled_xtrace")
                for trialid in trialid_list
            ]
        else:
            df_xtrace_list = [None] * len(trialid_list)
        dict_df["df_xtrace"] = df_xtrace_list

        if "relaxation_time" in list_read_dataname:
            df_relaxation_time_list = [
                load_trial(net_p, trialid, targetfile="relaxation_time")
                for trialid in trialid_list
            ]
        else:
            df_relaxation_time_list = [None] * len(trialid_list)
        dict_df["df_relaxation_time"] = df_relaxation_time_list

        if "MC" in list_read_dataname:
            df_MC_list = [
                load_trial(net_p, trialid, targetfile="MC") for trialid in trialid_list
            ]
        else:
            df_MC_list = [None] * len(trialid_list)
        dict_df["df_MC"] = df_MC_list
        ############################################################################

        if recursive_df:
            # df_net は いくつかのdfを入れ子にしたもの
            # trialのデータ(Sxとか)をdfとして入れ子dfを作る
            # DataFrameの作成がかなり重いっぽいので，データ量が多いときは注意
            df_net = pd.DataFrame(dict_df)
        else:
            # trialのデータ(Sxとか)を縦に結合, でかくなりすぎてクエリが遅くなるかも
            # df_net = pd.concat(df_saved_list, ignore_index=True)
            raise ValueError(
                "recursive_df=False is not implemented yet, please use recursive_df=True"
            )

        # df_netにネットワークのパラメタ情報と，trialの結果を1行にまとめて保存
        df_net = pd.concat(
            [df_net, pd.concat([df_net_params] * len(trialid_list), ignore_index=True)],
            axis=1,
        )

        df_net_list.append(df_net)

    if len(df_net_list) == 0:
        raise Exception("No net found")
    # ネットワークの種類 x トライル数 の列を持つDataFrameを作成
    df_sim = pd.concat(df_net_list, ignore_index=True)

    # デフォルトではstrがobjectとして読み込まれるので，stringへの変換
    for c in df_sim.columns:
        if isinstance(df_sim[c].loc[0], str):
            df_sim[c] = df_sim[c].astype("string")

    return df_sim


def load_sim_net():
    pass
