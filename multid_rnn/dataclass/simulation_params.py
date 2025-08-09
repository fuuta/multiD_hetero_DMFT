from dataclasses import dataclass

from serde import serialize

# Metric = Literal["whole_ys_trace", "idoled_ys_trace", "hetero_vars_info", "dys"]

# @serialize
# @dataclass(frozen=True)
# class SaveMetrics:
#     metrics: list[Metric]
#     Tidolrate: float

#     def __post__init__(self):
#         assert self.Tidolrate >= 0 and self.Tidolrate <= 1, "Tidolrate must be between 0 and 1"


@serialize
@dataclass(frozen=True)
class SimulationParameters:
    T: float
    dt: float
    seed: int
    n_trial: int
    n_block: int
    dir_prefix: str
    is_compute_mle: bool = False  # MLEを計算するかどうか
    is_ret_dys: bool = False  # dyを返すかどうか
    is_save_xtrace: bool = False  # xのトレースを保存するかどうか
    is_save_autocorr: bool = False  # 自己相関を保存するかどうか
    initstate_scale: float = 1e-1  # 初期状態のスケール

    def validate(self):
        assert self.T > 0, "T must be positive"
        assert self.dt > 0, "dt must be positive"
        assert self.n_trial > 0, "n_trial must be positive"
        assert self.n_block > 1, "n_block must be greater than 1"

    def __post__init__(self):
        self.validate()
