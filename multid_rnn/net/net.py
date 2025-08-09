# from analysis.mean_field import iterativemethod, tildeG
import traceback
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
import statsmodels.api as sm
from jax import lax, vmap

from ..dataclass.net_params import NetworkParameters
from ..dataclass.prob_dist import (
    GammaDist,
    LogNormalDist,
    NormalDist,
    TruncatedLogNormalDist,
    TruncatedNormalDist,
    TwoValDist,
    UniformDist,
)
from ..utils.logging_utils import get_logger

logger = get_logger()

sampling_approx_dist = (
    NormalDist,
    UniformDist,
    LogNormalDist,
    TruncatedNormalDist,
    TruncatedLogNormalDist,
    GammaDist,
)
n_samples = 100000  # 1000くらいだとかなりばらつくので, 100000くらいあるとなめらか (GPUメモリたくさん必要)


def critical_g_mdhetero_Muscinelli(net_params: NetworkParameters):
    hetero_info = net_params.hetero_info
    list_heterovars = 1  # ヘテロな変数は1個だけを想定
    assert len(net_params.coefficient) == 2, (
        "Coefficient must be 2D array for 2-variate model"
    )
    raise NotImplementedError("critical_g_mdhetero_Muscinelli is not implemented yet")


def get_average_f_Gf(
    net_params: NetworkParameters, rndkey: jax.Array, n_samples: int = n_samples
) -> tuple[Callable[[jnp.ndarray], jnp.ndarray], jax.Array]:
    hetero_info = net_params.hetero_info
    assert len(net_params.coefficient) == 2, (
        "Coefficient must be 2D array for 2-variate model"
    )
    assert hetero_info is not None

    def numerical_average_f_Gf(
        hetero_vals: jax.Array,
    ) -> Callable[[jnp.ndarray], jnp.ndarray]:
        if hetero_info.label == "gamma":
            beta = net_params.coefficient[1][0]
            # We want to map pure2varGf over the 'gamma' parameter, which comes from hetero_vals.
            # 'beta' is fixed, and 'omega' (freqRange) is an argument to f_average_Gf.
            # pure2varGf(gamma, beta, omega)
            # in_axes=(0, None, None) means:
            #   - The first argument to pure2varGf (gamma) is mapped from hetero_vals.
            #   - The second argument (beta) is treated as constant (broadcast).
            #   - The third argument (omega) is treated as constant (broadcast).
            vmapped_pure2varGf = vmap(pure2varGf, in_axes=(0, None, None))

            def f_average_Gf(freqRange: jnp.ndarray) -> jnp.ndarray:
                # hetero_vals provides the varying 'gamma'
                # beta is fixed
                # freqRange provides 'omega'
                return vmapped_pure2varGf(hetero_vals, beta, freqRange).mean(axis=0)

            return f_average_Gf
        elif hetero_info.label == "beta":
            gamma = -net_params.coefficient[1][1]  # 符号に注意
            # We want to map pure2varGf over the 'beta' parameter, which comes from hetero_vals.
            # 'gamma' is fixed, and 'omega' (freqRange) is an argument to f_average_Gf.
            # pure2varGf(gamma, beta, omega)
            # in_axes=(None, 0, None) means:
            #   - The first argument to pure2varGf (gamma) is treated as constant (broadcast).
            #   - The second argument (beta) is mapped from hetero_vals.
            #   - The third argument (omega) is treated as constant (broadcast).
            vmapped_pure2varGf = vmap(pure2varGf, in_axes=(None, 0, None))

            def f_average_Gf(freqRange: jnp.ndarray) -> jnp.ndarray:
                # gamma is fixed
                # hetero_vals provides the varying 'beta'
                # freqRange provides 'omega'
                return vmapped_pure2varGf(gamma, hetero_vals, freqRange).mean(axis=0)

            return f_average_Gf
        else:
            raise NotImplementedError()

    # def numerical_average_f_Gf(hetero_vals):
    #     if hetero_info.label == "gamma":
    #         beta = net_params.coefficient[1][0]
    #         def f_average_Gf(freqRange: jnp.ndarray):
    #             return jnp.asarray([pure2varGf(gamma=gamma, beta=beta, omega=freqRange) for gamma in hetero_vals]).mean(axis=0)
    #         return f_average_Gf
    #     elif hetero_info.label == "beta":
    #         gamma = net_params.coefficient[1][1]
    #         def f_average_Gf(freqRange: jnp.ndarray):
    #             return jnp.asarray([pure2varGf(gamma=gamma, beta=beta, omega=freqRange) for beta in hetero_vals]).mean(axis=0)
    #         return f_average_Gf

    if isinstance(hetero_info.dist, sampling_approx_dist):
        # 解析的に計算できないときはヘテロ平均を,サンプリングによって近似する
        hetero_vals, rndkey = hetero_info.dist.sample(rndkey=rndkey, shape=[n_samples])
        return numerical_average_f_Gf(hetero_vals), rndkey
    elif isinstance(hetero_info.dist, TwoValDist):
        # 2値分布のときは平均を直接計算する
        high_val = hetero_info.dist.high_val
        low_val = hetero_info.dist.low_val
        p = hetero_info.dist.p  # 低い値が選ばれる確率
        if hetero_info.label == "gamma":
            beta = net_params.coefficient[1][0]

            def f_average_Gf(freqRange: jnp.ndarray) -> jnp.ndarray:
                return p * pure2varGf(gamma=low_val, beta=beta, omega=freqRange) + (
                    1 - p
                ) * pure2varGf(gamma=high_val, beta=beta, omega=freqRange)

            return f_average_Gf, rndkey
        elif hetero_info.label == "beta":
            gamma = net_params.coefficient[1][1]

            def f_average_Gf(freqRange: jnp.ndarray) -> jnp.ndarray:
                return p * pure2varGf(gamma=gamma, beta=low_val, omega=freqRange) + (
                    1 - p
                ) * pure2varGf(gamma=gamma, beta=high_val, omega=freqRange)

            return f_average_Gf, rndkey
        else:
            raise ValueError("invalid hetero_info.label")
    else:
        raise ValueError("invalid hetero_info.dist type")


def critical_g_hetero2var(
    net_params: NetworkParameters, rndkey: jax.Array, n_samples: int = n_samples
):
    hetero_info = net_params.hetero_info
    list_heterovars = 1  # ヘテロな変数は1個だけを想定
    assert len(net_params.coefficient) == 2, (
        "Coefficient must be 2D array for 2-variate model"
    )

    f_Gf, rndkey = get_average_f_Gf(net_params, rndkey, n_samples=n_samples)

    freqRange = jnp.linspace(0, 10 * np.pi, 10000)
    Gf_vals = f_Gf(freqRange)
    argmax_f = freqRange[jnp.argmax(Gf_vals)]
    freqRange = jnp.linspace(argmax_f - np.pi, argmax_f + np.pi, 1001)
    Gf_vals = f_Gf(freqRange)  # 再度計算して,最大値の近傍を調べる
    Gmax = jnp.max(Gf_vals)
    critical_g = jnp.sqrt(1 / Gmax)
    return critical_g, f_Gf, rndkey


def pure2varGf(gamma, beta, omega):
    """
    GPA版のG(f; \gamma, \beta)
    ※ Adaptationとして使う時は, Adaptationの時のnetworkダイナミクスで定義される\betaと逆符号のものを入れること
    """
    return (omega**2 + gamma**2) / (
        omega**4
        + (gamma**2 + 1) * omega**2
        + gamma**2
        + beta**2
        + 2 * beta * omega**2
        - 2 * beta * gamma
    )


def calc_average_powerspectrum(xs, blockTime, dt, deltaT=1):
    """
    空間・時間の2次元信号xsから,各空間の次元に対して数値的にパワースペクトルを計算し,
    空間方向に平均化したパワースペクトルを返す

    Args:
        xs (_type_): [空間,時間]の次元を持つ信号
        blockTime (_type_): 時間の長さ
        dt (_type_): 時間刻みの大きさ
        deltaT (int, optional): 計算する周波数幅の係数(ナイキスト周波数を超えないために1以上である必要がある). Defaults to 1.

    Returns:
        _type_: _description_
    """
    deltaFreq = 1 / blockTime  # KHz
    maxFreq = 1.0 / (2.0 * dt * deltaT)  # KHz
    # freqRange = jnp.arange(-maxFreq, maxFreq, step=deltaFreq) #KHz これ間違いかも？
    freqRange = jnp.fft.fftshift(jnp.fft.fftfreq(xs[:, 0].size, d=dt))  #
    # SxB = jnp.zeros([freqRange.size, xs.shape[-1]], dtype=jnp.complex64)

    def compute_Sx_oneneuron(x_it):
        Fx_it = dt * deltaT * jnp.fft.fftshift(jnp.fft.fft(x_it))
        return jnp.abs(
            1.0 / (dt * deltaT * freqRange.size) * jnp.conjugate(Fx_it) * Fx_it
        )

    SxB = lax.map(compute_Sx_oneneuron, xs.T)
    return freqRange, jnp.mean(SxB, axis=0)


def calc_each_autocorrelation(xs, max_lag_T: float, dt: float, is_use_fft: bool = True):
    """
    空間・時間の2次元信号xsから,各空間の次元に対して数値的に自己相関関数を計算
    自己相関関数の計算にはstatsmodelsモジュールを使用

    """
    lags = np.linspace(0, max_lag_T, int(max_lag_T / dt))

    def compute_autocorr_oneneuron(x_it):
        return sm.tsa.acf(x_it, nlags=len(lags) - 1, fft=is_use_fft)

    ret = []
    for i in range(xs.shape[-1]):
        ret.append(np.expand_dims(compute_autocorr_oneneuron(xs.T[i, :]), -1))
    # Cs = lax.map(compute_autocorr_oneneuron, xs.T)
    return lags, np.concatenate(ret, -1)


def calc_Sx_ana(
    iterations,
    g,
    freqRange,
    nonlinearpass_type="analitycal",
    netParams=None,
    nlpassParams=None,
):
    raise NotImplementedError("calc_Sx_ana is not implemented yet")
    # if netParams is None:
    #     netParams = {
    #         "net_type": "heteroGPA1var",
    #         "mu_gamma": 10.0,
    #         "sigma_gamma": 0.0,
    #         "phi_type": "PWL",
    #     }
    # if nlpassParams is None:
    #     if nonlinearpass_type == "analitycal":
    #         nlpassParams = {"dIntSigmaFactor": 200}
    #     elif nonlinearpass_type == "mc":
    #         nlpassParams = {}

    # S0 = np.ones(freqRange.size, dtype=np.float64)
    # try:
    #     dump_iters, Sxs = iterativemethod(
    #         iterations,
    #         freqRange,
    #         S0,
    #         g,
    #         nonlinearpass_type,
    #         netParams,
    #         nlpassParams,
    #         verbose=0,
    #     )
    #     # dump_iters, Sxs = iterativemethod(50, freqRange, S0, g, nonlinearpass_type, netParams, nlpassParams, verbose=1, iter2dump=10)
    # except Exception:
    #     traceback.print_exc()
    # return dump_iters, Sxs
