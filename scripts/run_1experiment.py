import argparse
import json
import os
import pickle
import traceback
from pathlib import Path

import jax
import numpy as np
from serde.json import from_json
import pandas as pd
import gc

from multid_rnn.utils.mc import compute_MCtaus_regression
from multid_rnn.dataclass.net_params import NetworkParameters
from multid_rnn.dataclass.simulation_params import SimulationParameters
from multid_rnn.net.net import (
    calc_average_powerspectrum,
    calc_each_autocorrelation,
    critical_g_hetero2var,
)
from multid_rnn.simulation.trial_scan import run_1trial_ndhetero
from multid_rnn.utils.jax_setting import set_jax_device_config
from multid_rnn.utils.logging_utils import get_logger, init_execution_trace_logging
from multid_rnn.utils.lyap import calc_lyap_from_sim_v2
from multid_rnn.utils.vis import (
    vis_autocorr_each_neuron,
    vis_calclyap,
    vis_Sx_Sphi,
    vis_trace,
    vis_var_hist,
    vis_relaxation_time_each_hetero_param,
)
from multid_rnn.dataclass.external_input import NoiseSourceParams
from multid_rnn.utils.name import determine_experiment_dirname

logger = get_logger()


set_jax_device_config()


def run_trials(
    sim_params: SimulationParameters,
    net_params: NetworkParameters,
    seed: int,
    can_skip: bool = True,
):
    # region シミュレーションパラメータの取得
    T = sim_params.T
    dt = sim_params.dt
    n_timestep = int(T / dt)
    n_block = sim_params.n_block
    is_compute_mle = sim_params.is_compute_mle
    is_save_xtrace = sim_params.is_save_xtrace
    is_save_autocorr = sim_params.is_save_autocorr
    initstate_scale = sim_params.initstate_scale
    # endregion
    Tidolrate = 0.5

    # region ネットワークパラメータの取得
    N = net_params.n_neuron
    D = net_params._n_dimension
    hetero_info = net_params.hetero_info
    g = net_params.coupling_strength
    phi_type = net_params.activation_function
    phi = phi_type.to_f()
    eI_params = net_params.external_input
    if D is None:
        D = net_params._n_dimension
    # endregion

    # region 乱数シードの初期化
    rndkey = jax.random.PRNGKey(seed)
    np.random.seed(seed)
    strrndkey = "-".join([str(r) for r in rndkey])
    # endregion

    # region directory名の決定, skipの判定
    trial_dir = determine_experiment_dirname(
        sim_params=sim_params,
        net_params=net_params,
    )
    trial_dir.mkdir(exist_ok=True, parents=True)
    # If simulation data already exists, skip or rerun
    done_file = Path(trial_dir) / f"{strrndkey}.done"
    if done_file.exists() and can_skip:
        logger.info(f"SKIP: Trial {done_file}")
        return
    # endregion

    if not (trial_dir / "net_params.pkl").exists():
        with (trial_dir / "net_params.pkl").open("wb") as f:
            pickle.dump(net_params, f)

    # --------------------------------------------------------------------------------------------

    # region シミュレーションの本体
    logger.info("START: microscipic network simulation")
    assert n_timestep % n_block == 0

    # 1回trialを実行
    if n_timestep < 20000: 
        trial_result, rndkey = run_1trial_ndhetero(
            rndkey=rndkey,
            n_timestep=n_timestep,
            n_block=1,
            dt=dt,
            N=N,
            D=D,
            g=g,
            phi=phi,
            hetero_info=hetero_info,
            net_params=net_params,
            initstate_scale=initstate_scale,  # Use the scale from the parameters
            isret_dy=False,
        )
    else:
        n_block = min(n_block, int(n_timestep / 10000))  # 20000 step, 3000 2var neurons で8GBくらいのGPUメモリ, ブロック数が多すぎると逆に遅くなるので注意
        trial_result, rndkey = run_1trial_ndhetero(
            rndkey=rndkey,
            n_timestep=n_timestep,
            n_block=n_block,
            dt=dt,
            N=N,
            D=D,
            g=g,
            phi=phi,
            hetero_info=hetero_info,
            net_params=net_params,
            initstate_scale=initstate_scale,
            isret_dy=False,
        )
    logger.info("END: microscipic network simulation")
    # endregion

    # --------------------------------------------------------------------------------------------

    ys = np.asarray(trial_result.ys_history)
    idoled_ys = ys[int(Tidolrate * ys.shape[0]) :]

    # region hetero変数の可視化・保存
    if hetero_info is not None:
        hetero_var_vals = np.asarray(trial_result.hetero_var_vals)
        os.makedirs(trial_dir / "heteroparams", exist_ok=True)
        vis_var_hist(
            hetero_info.label,
            hetero_var_vals,
            hetero_info,
            statsinfo=None,
            path=trial_dir / "heteroparams" / f"{hetero_info.label}hist-{strrndkey}",
        )
        np.savez_compressed(
            trial_dir / f"hetero_var_vals-{strrndkey}.npz",
            hetero_vals=hetero_var_vals,
        )
        logger.info("END: hetero変数の可視化・保存")
    # endregion

    # region y(t), dy(t) の可視化・保存
    # 全時間のx(t)の可視化
    vis_trace(
        ys[:, :N],
        dt,
        title="",
        path=trial_dir / f"xtrace-{strrndkey}",
    )

    vis_trace(
        idoled_ys[:, :N],
        dt,
        title="",
        path=trial_dir / f"x_idoledtrace-{strrndkey}",
    )
    if is_save_xtrace:
        np.savez_compressed(
            trial_dir / f"xtrace-{strrndkey}.npz",
            xtrace=ys[:, :N],
            dt=dt,
        )
    logger.info("END: y(t), dy(t) の可視化・保存")
    # endregion

    #region 外部入力の可視化・MCの計算
    if eI_params is not None:
        sources = np.asarray(trial_result.eIsource_history)
        if isinstance(eI_params.soureparam, NoiseSourceParams):
            # 時間遅れeIの予測
            idoled_sources = sources[int(Tidolrate * sources.shape[0]) :]
            assert idoled_sources.shape[1] == 1
            max_lag_time = min(idoled_sources.shape[0]/2.*dt, 100.) # 半分まで or 100秒まで
            max_dense_lag = max_lag_time/5. # 1/5までは密にlagをとる
            lag_array = np.concat([np.linspace(0., max_dense_lag, 21), np.linspace(max_dense_lag+dt, max_lag_time, 21)])
            mc_dump_path = trial_dir / "MC"
            mc_dump_path.mkdir(parents=True, exist_ok=True)
            
            list_MCtau, totalMC = compute_MCtaus_regression(
                signals=np.asarray(idoled_ys[:, :N]),
                y = np.asarray(idoled_sources),
                lag_array=lag_array,
                dt=dt,
                trainrate=0.5
            )
            np.savez_compressed(
                mc_dump_path / f"regression-all{N}readout-{strrndkey}.npz",
                MCtaus=np.asarray(list_MCtau),
                totalMC=totalMC,
                lag_array=lag_array,
                dt=dt,
                n_readout=N,
                is_assume_0_correlation=True
            )
    #endregion

    # region x(t)から最大Lyapunov指数の計算
    if is_compute_mle:
        lyap_params = {
            "N": N,
            "D": D,
            "J": trial_result.J,
            "hetero_vals": trial_result.hetero_var_vals,
            "net_params": net_params,
            "phi_type": phi_type,
        }
        lyap_params["d0_init"] = {"type": "randnormal", "scale": 1e-6}
        t_rndkey = rndkey
        last_dval, scaled_d_norm, lambda_values, mean_lambda_values, rndkey = (
            calc_lyap_from_sim_v2(
                idoled_ys,
                t_rndkey,
                lyap_params,
                n_timestep=idoled_ys.shape[0],
                dt=dt,
                f_jacob=trial_result.f_jacob,
                norm_steps=100,
                verbose=0,
                n_block=n_block,
            )
        )
        vis_calclyap(
            scaled_d_norm,
            lambda_values,
            mean_lambda_values,
            trial_dir / f"lyap_randnormal-{strrndkey}",
        )

        lyap_params["d0_init"] = {"type": "value", "scale": 1e-6, "value": last_dval}
        last_dval, scaled_d_norm, lambda_values, mean_lambda_values, rndkey = (
            calc_lyap_from_sim_v2(
                idoled_ys,
                t_rndkey,
                lyap_params,
                n_timestep=idoled_ys.shape[0],
                dt=dt,
                f_jacob=trial_result.f_jacob,
                norm_steps=100,
                verbose=0,
                n_block=n_block,
            )
        )
        vis_calclyap(
            scaled_d_norm,
            lambda_values,
            mean_lambda_values,
            trial_dir / f"lyap_dT-{strrndkey}",
        )
        logger.debug("END: x(t)から最大Lyapunov指数の計算")

        np.savez_compressed(
            trial_dir / f"MLE-{strrndkey}.npz",
            maxlyap=mean_lambda_values[-1],
            maxlyapraw=lambda_values[-1],
        )

    # endregion

    # region x(t)からSxの計算, phi(x(t))からSphixの計算
    idoled_xs = idoled_ys[:, :N]
    if idoled_xs.shape[0] % 2 == 0:
        idoled_xs = idoled_xs[1:, :]
    freqRange, Sx = calc_average_powerspectrum(
        idoled_xs, idoled_xs.shape[0] * dt, dt, deltaT=1
    )
    freqRange, Sphix = calc_average_powerspectrum(
        phi(idoled_xs), idoled_xs.shape[0] * dt, dt, deltaT=1 # type: ignore
    )

    # 可視化：SxとSphix
    vis_Sx_Sphi(freqRange, Sx, Sphix, path=trial_dir / f"S-{strrndkey}")

    np.savez_compressed(
        trial_dir / f"SxSphix-{strrndkey}.npz",
        freqRange=freqRange,
        Sx=Sx,
        Sphix=Sphix,
    )
    # endregion

    # region x(t), a(t) から自己相関関数・緩和時間の計算
    # idolingした後のy(t)から自己相関関数を計算
    autocorrs_lags, autocorrs = calc_each_autocorrelation(
        idoled_ys, max_lag_T=idoled_ys.shape[0] * dt / 2.0, dt=dt
    )

    # 自己相関関数のnpz保存 (大容量)
    if is_save_autocorr:
        np.savez_compressed(
            trial_dir / f"autocorrs-{strrndkey}.npz",
            autocorrs=autocorrs, # 各ニューロンの自己相関関数, 容量が大きいので一旦無視
            autocorrs_lags=autocorrs_lags,
        )

    vis_autocorr_each_neuron(
        autocorrs_lags,
        autocorrs,
        path=trial_dir / f"autocorr_ys-{strrndkey}.png",
    )
    for d in range(D):
        vis_autocorr_each_neuron(
            autocorrs_lags,
            autocorrs[:, N * d : N * (d + 1)],
            path=trial_dir / f"autocorr_d{d}-{strrndkey}.png",
        )
        vis_autocorr_each_neuron(
            autocorrs_lags,
            np.expand_dims(autocorrs[:, N * d : N * (d + 1)].mean(-1), -1),
            path=trial_dir / f"autocorr_mean_d{d}-{strrndkey}.png",
        )

    # 緩和時間の計算
    # 自己相関関数の値がe^(-1)以下になる最初の時間遅れを緩和時間とする
    def find_first_index(autocorr, ref_value):
        if ref_value == -1.: # autocorrがnanの場合
            return np.nan
        idx = np.argmax(autocorr < ref_value)
        if (autocorr < ref_value).any():
            return idx
        else:
            return np.nan

    # ベクトル化
    v_find_first_index = np.vectorize(find_first_index, signature='(n),()->()', otypes=[float])
    ref_value = autocorrs[0, :] / np.exp(1) # 自己相関は0が最大(1), 最大値が1/eになる時間遅れを求める, nanがあり得る
    ref_value = np.nan_to_num(ref_value, nan=-1.0)  # NaNを-1.0に置き換える
    ref_indices = v_find_first_index(autocorrs.T, ref_value)
    relaxation_time = np.asarray([autocorrs_lags[int(ref_ind)] if not np.isnan(ref_ind) else np.nan for ref_ind in ref_indices])
    np.savez_compressed(
        trial_dir / f"relaxation_time-{strrndkey}.npz",
        relaxation_time=relaxation_time,
    )

    if hetero_info is not None:
        for d in range(D):
            vis_relaxation_time_each_hetero_param(
                relaxation_time=relaxation_time[N * d : N * (d + 1)],
                hetero_var_vals=trial_result.hetero_var_vals,
                hetero_var_label=hetero_info.label,
                path=trial_dir / f"relaxation_time_d{d}-{strrndkey}.png",
            )
        
    
    logger.info("END: x(t)から自己相関関数・緩和時間の計算")
    # endregion

    if eI_params is not None:
        np.savez_compressed(
            trial_dir / f"eI_params-{strrndkey}.npz",
            sources=np.asarray(trial_result.eIsource_history),
            Win=np.asarray(trial_result.Win),
        ) # 実際の入力は source @ Win.T になる.
        logger.info("END: eIの保存")

    
    # g_cとG(freqRange)を計算して保存
    if len(net_params.coefficient) == 2:
        g_c, f_Gf, rndkey = critical_g_hetero2var(net_params, rndkey, n_samples=5000) # かなりラフなサンプル平均で近似
    else:
        g_c, f_Gf = np.nan, None

    if not (trial_dir / "g_c.txt").exists():
        np.savetxt(trial_dir / "g_c.txt", np.expand_dims(g_c, -1))

    done_file.touch()  # 終了フラグ用の空ファイルを作成
    logger.info("END: whole trial {}".format(trial_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Run 1 Trial",
        description="run microscopic simulation of multi dimentinal neuron random neural network",
    )
    project_root = Path(
        os.environ.get("PROJECT_DIR", Path(__file__).resolve().parents[2])
    )
    parser.add_argument(
        "--sim_params_path",
        type=str,
        help="Path to the JSON file containing simulation parameters.",
    )
    parser.add_argument(
        "--net_params_path",
        type=str,
        help="Path to the JSON file containing network parameters.",
    )
    parser.add_argument(
        "--seed", type=int, help="seed of random number generator", default=1234
    )

    args = parser.parse_args()
    sim_params_path = args.sim_params_path
    net_params_path = args.net_params_path
    if sim_params_path is None or net_params_path is None:
        raise ValueError("sim_params_path and net_params_path must be provided")
    
    seed = args.seed
    can_skip = True

    execution_log_path = project_root / f"TEMP/logs/run_trial_seed{seed}.log"
    init_execution_trace_logging(logger, execution_log_path)

    if sim_params_path == "" or net_params_path == "":
        raise Exception("sim_params_path, net_params_path must be set")

    with Path(sim_params_path).open("r") as fp:
        sim_params_json = json.load(fp)
    with Path(net_params_path).open("r") as fp:
        net_params_json = json.load(fp)

    _sim_params = from_json(SimulationParameters, json.dumps(sim_params_json))
    _net_params = from_json(NetworkParameters, json.dumps(net_params_json))

    try:
        run_trials(_sim_params, _net_params, seed=seed, can_skip=can_skip)
    except Exception as e:
        t = list(traceback.TracebackException.from_exception(e).format())
        logger.error(t)
        logger.exception(f"{e}")
