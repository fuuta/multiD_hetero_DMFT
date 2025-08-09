import json
import os
from functools import partial
from pathlib import Path
from typing import Callable

import numpy as np
from numpy.random import default_rng
from serde.json import to_json

from multid_rnn.dataclass.simulation_params import SimulationParameters
from multid_rnn.dataclass.net_params import NetworkParameters
from multid_rnn.utils.json_encoder import NumpyJSONEncoder
from multid_rnn.utils.logging_utils import get_logger, init_execution_trace_logging
from multid_rnn.utils.run import run_trials_with_subprocess, check_can_skip, run_direct


def run_multi_experiments(
    sim_params: SimulationParameters,
    create_net_params: Callable[[int, float, float], NetworkParameters],  # (N, var, g) -> NetworkParameters を返す関数
    list_N: list[int],
    list_var: list[float],
    var_label: str,
    list_g: list[float],
    n_parallel: int = 2,
    expr_name: str = "defaults",  # 実験の名前
    is_run_direct: bool = False, # subprocessを使わずに直接実行する. Debug用
):
    logger = get_logger()
    project_root = Path(os.environ["PROJECT_DIR"])
    expr_path = project_root / "TEMP" / expr_name
    expr_path.mkdir(parents=True, exist_ok=True)

    init_execution_trace_logging(logger, expr_path / "execution.log")

    json_path = expr_path / "json"
    json_path.mkdir(parents=True, exist_ok=True)

    seed = sim_params.seed

    rg = default_rng(seed)  # 乱数生成器の初期化

    NN, VV, GG = np.meshgrid(
        list_N, list_var, list_g
    )  # list_Nとlist_pとlist_gの組み合わせ
    NN, VV, GG = NN.flatten(), VV.flatten(), GG.flatten()
    ii = np.argsort(NN)
    NN, VV, GG = NN[ii], VV[ii], GG[ii]  # Nが小さい順に並び替え

    # 各トライアルの初期シード (各トライアルでは異なるが, 異なる実験条件の同一トライアルでは同じシードを使用する)
    trial_seeds = rg.integers(0, 100000, size=sim_params.n_trial).tolist()

    skip_list = []  # スキップする実験条件のインデックスを格納するリスト

    # 実験条件に応じたjsonファイルをあらかじめ作成しておく
    with open(json_path / "temp_sim_params.json", "w") as fp:
        json.dump(json.loads(to_json(sim_params)), fp, indent=4, cls=NumpyJSONEncoder)
    for i in range(NN.size):
        g = GG[i].item()
        N = int(NN[i].item())
        v = VV[i].item()

        net_params = create_net_params(N, v, g)
        with open(json_path / f"temp_net_params_{i}.json", "w") as fp:
            json.dump(
                json.loads(to_json(net_params)), fp, indent=4, cls=NumpyJSONEncoder
            )

        # 全トライアルが存在していれば該当実験条件をスキップ
        if all(
            [
                check_can_skip(sim_params=sim_params, net_params=net_params, seed=seed)
                for seed in trial_seeds
            ]
        ):
            skip_list.append(i)

        if is_run_direct:
            logger.info("--- RUN DIRECT MODE (each 1 trial) ---")
            run_direct(sim_params, net_params, can_skip=False, seed=trial_seeds[0])
    if is_run_direct:
        exit(0)

    t_run_subp = partial(
        run_trials_with_subprocess,
        seeds=trial_seeds,
        n_parallel=n_parallel,
        logger=logger,
        json_path=json_path,
    )  # ある実験条件に対する関数を定義

    logger.info(
        "Starting parallel execution for %d experiments (%d will be skip) with %d trials each.",
        NN.size,
        len(skip_list),
        sim_params.n_trial,
    )
    # 各実験条件に対して並列で実行
    for i in range(NN.size):
        if i in skip_list:
            continue
        t_run_subp(i)
        logger.info(f"Experiment {(i + 1) / NN.size * 100}% Done")
