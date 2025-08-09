import matplotlib

matplotlib.use("Agg")
import traceback
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy
from .logging_utils import get_logger

logger = get_logger()


def save_plt_fig(fig, path: str | Path | None, postfix="", ext=".png"):
    if path is not None:
        if isinstance(path, Path):
            plt.savefig(path.with_suffix(ext))
        else:
            plt.savefig(path + postfix + ext)
        plt.close(fig)
    else:
        plt.show()
    plt.close("all")


def vis_trace_wrt_gamma(gammas, traces, dt, title):
    percentile90 = np.percentile(gammas, 99)
    indexs = gammas > percentile90
    plt.figure(figsize=[30, 10])
    ax = plt.subplot(2, 1, 1)
    vis_trace = traces[:, indexs]
    t = np.arange(0, vis_trace.shape[0]) * dt
    # for i in range(traces.shape[]):
    ax.plot(np.repeat(np.expand_dims(t, -1), vis_trace.shape[-1], axis=-1), vis_trace)
    # print(t.shape, traces.shape)
    # plt.plot(traces)
    # ax.set_xlabel("time")
    ax.set_ylabel(r"x_i(t)")
    ax.grid()
    # plt.xlim(0, 1000)
    ax.set_title("gamma_i > p90 = {}, {}".format(percentile90, title))

    # l_traces.append(traces)
    percentile10 = np.percentile(gammas, 1)
    indexs = gammas < percentile10
    # plt.figure(figsize=[30,10])
    ax = plt.subplot(2, 1, 2)
    vis_trace = traces[:, indexs]
    t = np.arange(0, vis_trace.shape[0]) * dt
    # for i in range(traces.shape[]):
    ax.plot(np.repeat(np.expand_dims(t, -1), vis_trace.shape[-1], axis=-1), vis_trace)
    # print(t.shape, traces.shape)
    # plt.plot(traces)
    ax.set_xlabel("time (s)")
    ax.set_ylabel(r"x_i(t)")
    ax.grid()
    # plt.xlim(0, 1000)
    ax.set_title("gamma_i < p10 = {}, {}".format(percentile10, title))


def vis_trace(
    traces, dt, title, vis_size=100, vis_index=None, x_label=r"x_i(t)", path=None
):
    N = traces.shape[-1]
    indexs = np.random.randint(0, N, size=vis_size) if vis_index is None else vis_index

    fig = plt.figure(figsize=[30, 5])
    ax = plt.subplot(1, 1, 1)
    vis_trace = traces[:, indexs]
    t = np.arange(0, vis_trace.shape[0]) * dt
    # for i in range(traces.shape[]):
    ax.plot(np.repeat(np.expand_dims(t, -1), vis_trace.shape[-1], axis=-1), vis_trace)
    # print(t.shape, traces.shape)
    # plt.plot(traces)
    # ax.set_xlabel("time")
    ax.set_ylabel(r"x_i(t)")
    ax.grid()
    # plt.xlim(0, 1000)
    ax.set_title("{}".format(title))

    save_plt_fig(fig, path)


def vis_tracehist(traces, title, xlabel, mode="1var", analytical_std=None, path=None):
    def normalpdfs(xmin, xmax, mu, sigma):
        dist = scipy.stats.norm
        kwargs = {"loc": mu, "scale": sigma}
        xs = np.linspace(xmin, xmax, 100)
        pdfs = dist.pdf(xs, **kwargs)
        return xs, pdfs

    if mode == "1var":
        N = traces.shape[-1]
    elif mode == "2var":
        N = int(traces.shape[-1] / 2)

    fig = plt.figure(figsize=[10, 10])
    ax = plt.subplot(1, 1, 1)

    if mode == "1var":
        # if x.shape[0] > 10000:
        #     x = scipy.signal.resample(x, num=10000, axis=0)
        x = traces.flatten()

        ax.hist(
            x,
            density=True,
            bins=100,
            histtype="step",
            alpha=1,
            label="normalized histogram",
        )

        xmin, xmax = ax.get_xlim()
        xs, pdfs = normalpdfs(xmin, xmax, np.mean(x), np.std(x))
        ax.plot(xs, pdfs, label=f"fitted normal N({np.mean(x)},{np.var(x)})")
        if analytical_std is not None:
            xs, pdfs = normalpdfs(xmin, xmax, 0, analytical_std)
            ax.plot(xs, pdfs, label=f"analytical normal N({0},{analytical_std**2})")
    else:
        ax.hist(
            traces[:, :N].flatten(),
            density=True,
            bins="auto",
            histtype="stepfilled",
            alpha=0.8,
            label="var1",
        )
        ax.hist(
            traces[:, N:].flatten(),
            density=True,
            bins="auto",
            histtype="stepfilled",
            alpha=0.8,
            label="var2",
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel("density")
    ax.grid()
    # plt.xlim(0, 1000)
    ax.set_title("{}".format(title))
    ax.legend()

    save_plt_fig(fig, path)


def vis_calclyap(scaled_d_norm, lambda_values, mean_lambda_values, path=None):
    try:
        fig = plt.figure()
        plt.plot(scaled_d_norm)
        plt.yscale("log")
        plt.title(
            "scaled delta norm: max{}, min{}".format(
                np.max(scaled_d_norm), np.min(scaled_d_norm)
            )
        )
        plt.grid()
        if path is None:
            plt.show()
        else:
            plt.savefig(path + "_distt.png")
            plt.close(fig)
    except Exception:
        traceback.print_exc()

    fig = plt.figure()
    plt.plot(lambda_values, c="black", alpha=0.5)
    plus_index = np.where(lambda_values >= 0)[0]
    minus_index = np.where(lambda_values <= 0)[0]
    plt.scatter(plus_index, lambda_values[plus_index], c="red", s=5, label=">=0")
    plt.scatter(minus_index, lambda_values[minus_index], c="blue", s=5, label="<0")
    plt.title(
        "temporal lambda: max{}, min{}, \n lastvalue{}".format(
            np.max(lambda_values), np.min(lambda_values), lambda_values[-1]
        )
    )
    plt.legend()
    plt.grid()

    save_plt_fig(fig, path, postfix="_lambdat")

    fig = plt.figure()
    plt.plot(mean_lambda_values, c="black", alpha=0.5)
    plus_index = np.where(mean_lambda_values >= 0)[0]
    minus_index = np.where(mean_lambda_values <= 0)[0]
    plt.scatter(plus_index, mean_lambda_values[plus_index], c="red", s=5, label=">=0")
    plt.scatter(minus_index, mean_lambda_values[minus_index], c="blue", s=5, label="<0")
    plt.title(
        "mean before all lambda: max{}, min{} \n lastvalue{}".format(
            np.max(mean_lambda_values),
            np.min(mean_lambda_values),
            mean_lambda_values[-1],
        )
    )
    plt.legend()
    plt.grid()

    save_plt_fig(fig, path, postfix="_meanlambdat")


def vis_Sx_Sphi(freqRange, Sx, Sphix, path=None):
    fig = plt.figure(figsize=[10, 10])
    plt.plot(freqRange, Sx, label="Sx(f)")
    plt.plot(freqRange, Sphix, label="Sphix(f)")
    plt.yscale("log")
    plt.legend()
    plt.xlabel("freq")
    plt.ylabel("power")
    plt.grid()
    plt.xlim(0, np.max(freqRange) * 0.2)
    df = freqRange[1] - freqRange[0]
    intSx = np.trapz(Sx, x=freqRange)
    intSphi = np.trapz(Sphix, x=freqRange)
    flag = "=" if intSx == intSphi else None
    if flag is None:
        flag = ">" if intSx > intSphi else "<"
    plt.title("int Sx(f)df ={} {} int Sphi(f)df ={}".format(intSx, flag, intSphi))

    save_plt_fig(fig, path, postfix="_SxSphix")


def vis_autocorr_each_neuron(autocorrs_lags, autocorrs, path: Path):
    fig = plt.figure()
    for i in range(autocorrs.shape[-1]):
        plt.plot(autocorrs_lags, autocorrs[:, i])
    plt.grid()
    plt.xlabel("lag")
    plt.ylabel("autocorrelation")
    plt.title("autocorrelation of each neuron")
    save_plt_fig(fig, path=path)


def vis_relaxation_time_each_hetero_param(
    relaxation_time, hetero_var_vals, hetero_var_label: str, path: Path
):
    fig = plt.figure()
    plt.scatter(hetero_var_vals, relaxation_time, marker="o", s=20, alpha=0.5)
    plt.xlabel(f"{hetero_var_label}")
    plt.ylabel("relaxation time")
    plt.grid()
    save_plt_fig(fig, path=path)


def vis_phi(phi):
    x = np.linspace(-2, 2, 1000)
    plt.figure()
    plt.plot(x, phi(x))
    plt.grid()
    plt.show()


def vis_analytical_Sxs(freqRange, dump_iters, Sxs, path=None):
    fig = plt.figure()
    for i, iters in enumerate(dump_iters):
        plt.plot(freqRange, Sxs[i], label="{}-th iter".format(iters))
    plt.yscale("log")
    plt.legend()
    plt.xlabel("freq")
    plt.ylabel("power")
    plt.grid()
    plt.xlim(0, np.max(freqRange) * 0.2)

    save_plt_fig(fig, path, postfix="_Sxs")


def vis_Sx_anaSx(freqRange, Sxs, anaSx, path=None):
    fig = plt.figure()
    for i in range(len(Sxs)):
        plt.plot(freqRange, Sxs[i], label="{}-th trial".format(i), linestyle="--")
    if anaSx is not None:
        plt.plot(freqRange, anaSx, label="analytical")
    plt.yscale("log")
    plt.legend()
    plt.xlabel("freq")
    plt.ylabel("power")
    plt.grid()
    plt.xlim(0, np.max(freqRange) * 0.2)

    save_plt_fig(fig, path, postfix="_SxanaSx")


def vis_var_hist(varname, varvals, dist_args, statsinfo=None, path=None):
    fig, ax = plt.subplots(1, 1)
    ax.hist(
        varvals,
        density=True,
        bins="auto",
        histtype="stepfilled",
        alpha=0.4,
        label="sampled hist",
    )
    if statsinfo is not None:
        if statsinfo.get("xs") is not None and statsinfo.get("pdfs") is not None:
            ax.plot(statsinfo["xs"], statsinfo["pdfs"], "--", alpha=0.8, label="pdf")
        if statsinfo.get("mean") is not None:
            ax.scatter(statsinfo["mean"]["x"], statsinfo["mean"]["Px"], label="mean")
        if statsinfo.get("median") is not None:
            ax.scatter(
                statsinfo["median"]["x"], statsinfo["median"]["Px"], label="median"
            )
        if statsinfo.get("mode") is not None:
            ax.scatter(statsinfo["mode"]["x"], statsinfo["mode"]["Px"], label="mode")
    ax.set_xlabel(varname)
    ax.set_title("{} distribution in this Network \n {}".format(varname, dist_args))
    ax.legend()
    save_plt_fig(fig, path)


def vis_trace_y_pred(traces, y, pred, dt, vis_size=100, title="", path=None):
    vis_length = 1000
    t = np.arange(0, traces.shape[-1]) * dt
    t = t[-vis_length:]
    traces = traces[:, -vis_length:]
    y = y[-vis_length:]
    pred = pred[-vis_length:]

    fig = plt.figure(figsize=[30, 15])
    ax = plt.subplot(3, 1, 1)

    ax.plot(np.tile(t, (vis_size, 1)).T, traces[:vis_size, :].T)
    ax.grid()
    ax.set_xlabel("time (s)")
    ax.set_ylabel(r"trace(t)")

    ax = plt.subplot(3, 1, 2)
    ax.plot(t, y, label="gt")
    ax.plot(t, pred, label="pred")
    ax.set_xlabel("time (s)")
    ax.set_ylabel(r"y(t)")
    ax.legend()
    ax.grid()

    ax = plt.subplot(3, 1, 3)
    ax.plot(t, np.sign(y), label="gt sign", alpha=0.9)
    ax.plot(t, np.sign(pred), label="pred sign", alpha=0.9)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("sign")
    ax.legend()
    ax.grid()

    acc = np.count_nonzero(np.sign(y) == np.sign(pred)) / y.size
    ax.set_title(f"sign acc={acc}")

    fig.suptitle("{}".format(title))

    save_plt_fig(fig, path)


def visMCk_acc(lag_array, dt, trainscore, testscore, trainacc, testacc, path=None):
    fig, ax1 = plt.subplots(figsize=[10, 10])
    ax2 = ax1.twinx()
    ax1.plot(lag_array * dt, trainscore, label="train score", color="blue")
    ax1.plot(lag_array * dt, testscore, label="test score", color="blue", linestyle=":")
    ax1.set_ylim(min(np.min(trainscore), np.min(testscore)), 1)
    ax1.grid()
    ax1.set_ylabel("score", color="blue")

    ax2.plot(lag_array * dt, trainacc, label="train acc", color="red")
    ax2.plot(lag_array * dt, testacc, label="test acc", color="red", linestyle=":")
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("accuracy (sign)", color="red")
    handler1, label1 = ax1.get_legend_handles_labels()
    handler2, label2 = ax2.get_legend_handles_labels()

    # 凡例をまとめて出力する
    ax1.legend(handler1 + handler2, label1 + label2, loc=2, borderaxespad=0.0)

    save_plt_fig(fig, path)


def vistotalMC(df, gs, path=None):
    fig = plt.figure(figsize=[10, 5])

    plt.errorbar(
        gs,
        [df[df["g"] == g]["MC"].mean() for g in gs],
        [df[df["g"] == g]["MC"].std() for g in gs],
        label="trainMC",
        alpha=0.75,
    )
    plt.scatter(df["g"], df["MC"], label="trainMC", marker="o", alpha=0.75)
    plt.legend()
    plt.grid()

    save_plt_fig(fig, path)


def vis_MCtaus(
    MCtaus: np.ndarray,
    totalMC: float,
    lag_array: np.ndarray,
    label: str,
    path: Path | None = None,
    canvas_ax: plt.Axes | None = None,
):
    if canvas_ax is None:
        fig, ax = plt.subplots(1, 1)
    else:
        ax = canvas_ax
    label += f"\n totalMC={totalMC}"
    ax.plot(lag_array, MCtaus, label=label)
    if canvas_ax is None:
        ax.grid()
        ax.set_xlabel("time lag (sec)")
        ax.set_ylabel("Memory Capacity")
        ax.legend()
        save_plt_fig(fig, path)


def plot_g_c_curve(
    seed: int,
    list_var: list[float],
    list_g: list[float],
    create_net_params: callable,
    dir_path: Path,
    var_label: str,
    n_samples: int = 100000
):
    # create_net_paramsは, p, N, gを引数にとり、NetworkParametersを返す関数
    import jax

    from multid_rnn.net.net import critical_g_hetero2var

    if not dir_path.exists():
        dir_path.mkdir(parents=True, exist_ok=True)

    rndkey = jax.random.PRNGKey(seed)  # JAXの乱数生成器の初期化
    list_g_c = []
    fig = plt.figure()
    for v in list_var:
        net_params = create_net_params(var=v, N=1, g=1.0) # Nやgはダミー
        g_c, _, rndkey = critical_g_hetero2var(net_params, rndkey, n_samples=n_samples)
        list_g_c.append(g_c)
        if len(list_g) > 0:
            plt.scatter(
                np.repeat(v, len(list_g)),
                list_g,
                color="black",
                label="simulated params",
            )

    p_gc = np.vstack([list_var, list_g_c]).T
    plt.plot(list_var, list_g_c, label=r"$g_c$", color="red")
    plt.xlabel(var_label)
    plt.ylabel(r"$g$")
    plt.grid()
    save_plt_fig(fig, dir_path / f"{var_label}_simulated_g_gc.png")
    # np.savetxt(dir_path / f"{var_label}_gc.txt", p_gc)