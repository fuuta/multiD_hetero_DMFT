import os
from pathlib import Path
from typing import Sequence

import numpy as np
import jax
import jax.numpy as jnp
import pandas as pd
import scipy
import scipy.integrate
from sklearn.linear_model import Lasso, LinearRegression, Ridge

from .logging_utils import get_logger

logger = get_logger()

def _traintest_split(
    x: np.ndarray, y: np.ndarray, trainrate: float = 0.8, is_shuffle: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    i = int(y.size * trainrate)
    if is_shuffle:
        new_ind = np.argsort(np.random.normal(size=[y.size]))
        x = x[new_ind, :]
        y = y[new_ind, :]
    trainx = x[:i, :]
    testx = x[i:, :]
    trainy = y[:i, :]
    testy = y[i:, :]
    return trainx, testx, trainy, testy

def _compute_MCtau_regression(
    tau: float,  # 時間遅れ (time sec)
    x: np.ndarray,  # 回帰元となる信号
    y: np.ndarray,  # 教師となる信号
    dt: float,  # シミュレーションで用いた時間刻み
    trainrate: float = 0.5,  # 訓練データの割合
    clfType: str = "Ridge",  # 線形分類器の種類を指定, "Linear", "Ridge", "Lasso"のいずれか
    fit_intercept: bool = False,  # 線形分類器の切片を学習するかどうか
    max_n_train_data: int = 10000,
):
    lag = int(tau / dt)
    assert lag >= 0, "lag must be >= 0"
    if lag == 0:
        lagged_x = x
        lagged_y = y
    else:
        lagged_x = x[lag:, :]
        lagged_y = y[:-lag, :]
    trainx, testx, trainy, testy = _traintest_split(
        lagged_x, lagged_y, trainrate=trainrate
    )

    if clfType == "Linear":
        clf = LinearRegression(fit_intercept=fit_intercept)
    elif clfType == "Ridge":
        clf = Ridge(fit_intercept=fit_intercept, alpha=1)
    elif clfType == "Lasso":
        clf = Lasso(fit_intercept=fit_intercept)
    else:
        raise ValueError(
            "clfType must be one of 'Linear', 'Ridge', or 'Lasso', got {}".format(
                clfType
            )
        )
    # データ数が多すぎると学習に時間がかかりすぎるので, ある程度間引く
    if trainx.shape[0] >= max_n_train_data:
        trainx = trainx[:max_n_train_data, :]
        trainy = trainy[:max_n_train_data, :]
    clf.fit(trainx, trainy)  # 分類器は訓練データに対して学習

    test_pred = clf.predict(testx)

    # NOTE:
    # clf.scoreは決定係数，本来のMCkは相関係数の2乗，最小二乗法の最適解を用いると決定係数と相関係数の2乗は一致する
    # しかし有限訓練データなので，分類器が過学習してしまうことを防ぐためにテストデータにおける係数を計算したい
    # この場合決定係数は負を取りうるので，時間遅れについて総和を取るMCが収束しない問題が起こる
    # よってここでは"テストデータ"に対する"相関係数の2乗"をMC_kとする

    MCtau_test = (
        np.corrcoef(np.squeeze(test_pred), np.squeeze(testy))[0, 1] ** 2
    )
    return MCtau_test

def compute_MCtaus_regression(
    signals: np.ndarray,  # 回帰元となる信号
    y: np.ndarray,  # 教師となる信号
    lag_array: np.ndarray,  # 時間遅れの配列
    dt: float,  # シミュレーションで用いた時間刻み
    trainrate: float,
    # path: Path | None = None,
):
    # 時間遅れごとに, 線形分類器を学習し, テストデータとの相関係数の2乗を記憶容量として計算する関数
    # 訓練データのサイズ (データ数, 入力次元数) が大きくなると計算量的にかなり重いので注意
    np_signals = np.asarray(signals)
    np_y = np.asarray(y)

    MCtaus = []
    for lag_val in lag_array:
        MCtaus.append(
            _compute_MCtau_regression(lag_val, np_signals, np_y, dt, trainrate=trainrate)
        )

    total_MC = scipy.integrate.trapezoid(
        MCtaus, lag_array
    )  # MCは時間遅れの総和で定義されるので，台形則で近似する
    return list(MCtaus), total_MC
