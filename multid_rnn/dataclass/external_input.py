from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp
import jax.random as jr

from ..utils.logging_utils import get_logger
from .prob_dist import NormalDist, ProbabilityDistribution, UniformDist

logger = get_logger()


@dataclass(frozen=True)
class WinParams:
    mix_type: Literal["same", "direct", "normal", "given"]
    prob_dist: ProbabilityDistribution | None
    Nsource: int  # 入力源の数 (Winの列数)
    p_in: float | None = None  # 入力結合の確率 (疎にする場合)
    nonzero_index: list[int] | None = None  # 入力結合の非ゼロインデックス (疎にする場合)
    given_Win: jnp.ndarray | None = None  # 与えられた結合行列 (mix_typeが"given"の場合に使用)

    def create(
        self,
        rndkey: jax.Array,
        N: int,  # 入力を入れたい変数の数 (一般的にはニューロンの数)
    ) -> tuple[jnp.ndarray, jax.Array]:
        mixtype = self.mix_type
        pdparam = self.prob_dist

        if mixtype == "same":  # ニューロンに同じ入力が与えられる
            assert self.Nsource == 1, "Nsource must be 1 for same mixing"
            assert pdparam is None, "pdparam must be None for same mixing"
            if self.nonzero_index is None: # 非ゼロインデックスが指定されていない場合は全結合
                Win = jnp.ones([N, 1])
            else:
                assert len(self.nonzero_index) > 0, "nonzero_index must be specified for same mixing"
                Win = jnp.zeros([N, 1])
                Win = Win.at[self.nonzero_index, 0].set(1.0)  # 非ゼロインデックスに1をセット 
        elif mixtype == "direct":  # 直接結合 ノイズ源がそのままニューロンに入力される
            assert self.Nsource == N, "Nsource must be N for direct mixing"
            assert pdparam is None, "pdparam must be None for direct mixing"
            Win = jnp.eye(N)
        elif (
            mixtype == "normal"
        ):  # 正規分布に従う結合でノイズが重み付き和されたものがニューロンに入力される
            assert isinstance(pdparam, NormalDist), (
                "pdparam must be NormalDist for normal mixing"
            )
            assert self.Nsource == N or self.Nsource == 1, (
                "Nsource must be N or 1 for normal mixing"
            )
            Win, rndkey = pdparam.sample(rndkey, shape=[N, self.Nsource])
        elif mixtype == "given":  # 与えられた結合行列を使用する
            assert self.given_Win is not None, "given_Win must be specified for given mixing"
            assert self.given_Win.shape == (N, self.Nsource), (
                f"given_Win must have shape ({N}, {self.Nsource})"
            )
            assert self.prob_dist is None, "prob_dist must be None for given mixing"
            assert self.p_in is None, "p_in must be None for given mixing"
            Win = self.given_Win
        else:
            raise Exception("invalid WinParams._type, use 'same', 'direct' or 'normal'")

        # 入力結合を疎にする場合
        if self.p_in is not None:
            # TODO: fan-inの確率を考慮した正規化
            is_connect = jr.uniform(rndkey, [N, self.Nsource]) < self.p_in
            rndkey, subkey = jax.random.split(rndkey)
            Win = jnp.multiply(Win, is_connect)

        # # 入力結合は常に正とする (!!!注意!!!)
        # Win = jnp.abs(Win)
        return Win, rndkey


@dataclass(frozen=True)
class BaseExternalInputSourceParams(ABC):
    Nsource: int

    @abstractmethod
    def generate(
        self, rndkey, start_ts: int, end_ts: int
    ) -> tuple[jnp.ndarray, jax.Array]: ...


@dataclass(frozen=True)
class WhiteNoiseSourceParams(BaseExternalInputSourceParams):
    sigma: float  # 拡散係数
    dt: float

    def generate(
        self, rndkey, start_ts: int, end_ts: int
    ) -> tuple[jnp.ndarray, jax.Array]:
        sources, rndkey = NormalDist(mean=0.0, sigma=1.0).sample(
            rndkey, shape=[end_ts - start_ts, self.Nsource]
        )
        return (
            sources * jnp.sqrt(2 * self.dt) * self.sigma / self.dt,
            rndkey,
        )  # NOTE: 実装ではeIに必ずdtがかかるので, あらかじめdtで割ってsqrt(dt)をかけておく, 2倍は拡散係数との兼ね合いあってもなくてもいい
        
    
@dataclass(frozen=True)
class UniformNoiseSourceParams(BaseExternalInputSourceParams):
    sigma: float
    dt: float

    def generate(
        self, rndkey, start_ts: int, end_ts: int
    ) -> tuple[jnp.ndarray, jax.Array]:
        d = UniformDist(min_val=-3**(1/2), max_val=3**(1/2))  # 平均0分散1の正規分布
        sources, rndkey = d.sample(
            rndkey, shape=[end_ts - start_ts, self.Nsource]
        )
        return (
            sources * jnp.sqrt(self.dt) * self.sigma / self.dt,
            rndkey,
        )  # NOTE: 実装ではeIに必ずdtがかかるので, あらかじめdtで割ってsqrt(dt)をかけておく


@dataclass(frozen=True)
class OUSourceParams(BaseExternalInputSourceParams):
    mu: float  # 平均値
    theta: float  # 減衰率
    sigma: float  # 拡散係数
    dt: float  # OU過程が進行する時間刻み
    init_source: jnp.ndarray | None

    def set_init_source(self, init_source: jnp.ndarray | None):
        object.__setattr__(self, "init_source", init_source)

    def generate(self, rndkey, start_ts: int, end_ts: int):
        if self.init_source is None:
            self.set_init_source(jnp.zeros(shape=[self.Nsource]))
        ou_source, rndkey = NormalDist(mean=0.0, sigma=1.0).sample(
            rndkey, shape=[end_ts - start_ts, self.Nsource]
        )

        noise = jnp.zeros_like(ou_source)  # OU過程の初期化 (状態は0)
        if self.init_source is not None:  # 初期値が与えられている場合はそれを使う
            noise = noise.at[0, :].set(self.init_source)
        mu = self.mu
        theta = self.theta
        sigma = self.sigma
        dt = self.dt
        # OU過程のシュミレーション (euler-Maruyama法)
        for i in range(1, end_ts - start_ts):
            dnoise = (
                -theta * (noise[i - 1, :] - mu) * dt
                + jnp.sqrt(2 * sigma * dt) * ou_source[i - 1, :]
            )
            noise = noise.at[i, :].set(noise[i - 1, :] + dnoise)
        return noise, rndkey


@dataclass(frozen=True)
class PulseSourceParams(BaseExternalInputSourceParams):
    start_T: float  # 開始時間 
    end_T: float  # 終了時間 
    length_T: float  # パルスの長さ 
    interval_T: float  # パルスの間隔 
    amp: float
    dt: float

    @property
    def start_ts(self):
        return int(self.start_T / self.dt)

    @property
    def end_ts(self):
        return int(self.end_T / self.dt)

    @property
    def length_ts(self):
        return int(self.length_T / self.dt)

    @property
    def interval_ts(self):
        return int(self.interval_T / self.dt)

    def on_off_time(self):
        list_on_off_time = [jnp.zeros(shape=[self.start_ts])]
        is_on = True  # パルスがオンの状態かどうか
        counter = self.start_ts
        while counter < self.end_ts:
            if is_on:
                if counter + self.length_ts <= self.end_ts:
                    # パルスの長さがend_tsを超えない場合
                    list_on_off_time.append(jnp.ones(shape=[self.length_ts]))
                    counter += self.length_ts
                else:
                    # パルスの長さがend_tsを超える場合は、end_tsまでセット
                    list_on_off_time.append(jnp.ones(shape=[self.end_ts - counter]))
                    counter = self.end_ts
                is_on = False
            else:
                if counter + self.interval_ts <= self.end_ts:
                    # パルスの間隔がend_tsを超えない場合
                    list_on_off_time.append(jnp.zeros(shape=[self.interval_ts]))
                    counter += self.interval_ts
                else:
                    # パルスの間隔がend_tsを超える場合は、end_tsまでセット
                    list_on_off_time.append(jnp.zeros(shape=[self.end_ts - counter]))
                    counter = self.end_ts
                is_on = True
        # パルスのオンオフ時間を結合
        return jnp.concatenate(list_on_off_time, axis=0)

    def generate(self, rndkey, start_ts: int, end_ts: int):
        on_off_time = self.on_off_time()
        tiled_on_off_time = jnp.tile(on_off_time[:, jnp.newaxis], (self.Nsource, 1))
        ret = tiled_on_off_time[start_ts:end_ts, :]*self.amp
        if ret.shape[0] < end_ts - start_ts:
            ret = jnp.concat([ret, jnp.zeros(shape=[end_ts - start_ts - ret.shape[0], self.Nsource])], axis=0)
        return ret, rndkey


@dataclass(frozen=True)
class SinewaveSourceParams(BaseExternalInputSourceParams):
    freq: float = jnp.pi / 100  # default 0.01Hz
    amp: float = 1.0
    phase: float = 0.0
    dt: float = 1.0

    def generate(self, rndkey, start_ts: int, end_ts: int):
        assert self.Nsource == 1, "Nsource must be 1 for sinewave input"
        source = jnp.zeros(shape=[end_ts - start_ts, self.Nsource])
        t = jnp.arange(start_ts, end_ts) * self.dt
        source = source.at[:, 0].set(
            self.amp * jnp.sin(2 * jnp.pi * self.freq * t + self.phase)
        )
        return source, rndkey


SourceDistParams = (
    WhiteNoiseSourceParams | OUSourceParams | PulseSourceParams | SinewaveSourceParams
)

NoiseSourceParams = WhiteNoiseSourceParams | OUSourceParams


@dataclass(frozen=True)
class ExternalInputParams:
    Win_params: WinParams # 入力結合のパラメータ
    soureparam: SourceDistParams | list[SourceDistParams]  # 入力源のパラメータ, listにするとsourceが足される
    is_only_input_to_x: bool = True

    def generate(
        self,
        rndkey,
        Win: jnp.ndarray,
        start_ts: int,
        end_ts: int,
        N: int,
        D: int,
    ):
        if self.is_only_input_to_x:
            assert Win.shape[0] == N, "Win shape mismatch with N"
        else:
            assert Win.shape[0] == N * D, "Win shape mismatch with N*D"
        if isinstance(self.soureparam, list):
            for i in range(len(self.soureparam)):
                assert Win.shape[1] == self.soureparam[i].Nsource, (
                    "Win shape mismatch with Nsource"
                )
        else:
            assert Win.shape[1] == self.soureparam.Nsource, (
                "Win shape mismatch with Nsource"
            )
        n_timestep = end_ts - start_ts

        if isinstance(self.soureparam, list):
            # 複数の入力源がある場合はそれぞれのsourceを生成して足し合わせる
            source_list = []
            for i, param in enumerate(self.soureparam):
                source, rndkey = param.generate(rndkey, start_ts, end_ts)
                source_list.append(source)
            source = jnp.expand_dims(jnp.sum(jnp.concat(source_list, axis=1), axis=1), -1)  # 各sourceを足し合わせる
        else:
            source, rndkey = self.soureparam.generate(
                rndkey, start_ts=start_ts, end_ts=end_ts
            )
        assert source.shape[0] == n_timestep, (
            f"source shape {source.shape} mismatch with n_timestep {n_timestep} ({end_ts} - {start_ts})"
        )

        eI = source @ Win.T
        # TODO: optimal sequenceの定義するノイズ

        if self.is_only_input_to_x:
            eI = jnp.hstack(
                [eI, jnp.zeros([n_timestep, N * D - N])]
            )  # インプットはxにのみ与える
        else:
            # TODO: x以外の入力を与える場合の処理
            raise NotImplementedError()

        return rndkey, eI, source
