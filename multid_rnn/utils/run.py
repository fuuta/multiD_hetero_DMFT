import os
import subprocess
import time
from pathlib import Path
import logging
import jax
from .name import determine_experiment_dirname
from ..dataclass.simulation_params import SimulationParameters
from ..dataclass.net_params import NetworkParameters
from .logging_utils import get_logger

CUDA_DEVICE_POOL = ["3", "2"]
logger = get_logger()


def check_can_skip(
    sim_params: SimulationParameters, net_params: NetworkParameters, seed: int
):
    rndkey = jax.random.PRNGKey(seed)
    strrndkey = "-".join([str(r) for r in rndkey])
    # endregion

    # region directory名の決定, skipの判定
    trial_dir = determine_experiment_dirname(
        sim_params=sim_params,
        net_params=net_params,
    )
    # If simulation data already exists, skip or rerun
    done_file = Path(trial_dir) / f"{strrndkey}.done"
    if done_file.exists():
        # logger.info(f"SKIP: Trial {done_file}")
        return True
    else:
        return False


def run_trials_with_subprocess(
    experiment_index: int,
    seeds: list[int],
    logger: logging.Logger,
    json_path: Path,
    n_parallel: int = 1,
):
    pcounter = 0
    processes = []
    exit_codes_history = []
    for i_trial in range(len(seeds)):
        my_env = os.environ.copy()
        my_env["CUDA_VISIBLE_DEVICES"] = CUDA_DEVICE_POOL[
            i_trial % len(CUDA_DEVICE_POOL)
        ]
        process = subprocess.Popen(
            [
                "uv",
                "run",
                "python",
                "scripts/run_1experiment.py",
                "--sim_params_path",
                json_path / "temp_sim_params.json",
                "--net_params_path",
                json_path / f"temp_net_params_{experiment_index}.json",
                "--seed",
                f"{seeds[i_trial]}",
            ],
            env=my_env,
            text=True,  # False(Default): バイトとして返す, True: stringとして返す
        )  # process instanceを返す
        time.sleep(0.01)
        processes.append(process)
        pcounter += 1
        if pcounter == n_parallel:  # n_parallel個のプロセスを立ち上げたら
            exit_codes = [
                p.wait() for p in processes
            ]  # 全てのプロセスが終了するまで待つ
            exit_codes_history += exit_codes
            processes = []  # プロセスリストをリセット
            pcounter = 0  # カウンターをリセット
        elif i_trial == len(seeds) - 1:  # 最後のトライアルの場合
            exit_codes = [
                p.wait() for p in processes
            ]
            exit_codes_history += exit_codes
    logger.debug(
        "experiment_index: %d, exit_codes: %s", experiment_index, exit_codes_history
    )


def run_direct(sim_params, net_params, can_skip=False, seed=1234):
    from scripts.run_1experiment import run_trials

    run_trials(sim_params, net_params, can_skip=can_skip, seed=seed)
