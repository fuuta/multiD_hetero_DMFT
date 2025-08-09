import os

import matplotlib

matplotlib.use("Agg")
import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import tqdm
import matplotlib.pyplot as plt


from multid_rnn.utils.data import SAVED_DATA_TYPES, load_net
from multid_rnn.utils.jax_setting import set_jax_device_config
from multid_rnn.utils.json_encoder import NumpyJSONEncoder
from multid_rnn.utils.logging_utils import get_logger
from multid_rnn.utils.vis_parameter_space import (
    concat_figdata,
    plot2d_heatmap,
    plot3d_surface,
    plot_gc_line,
    save_mofify_fig,
    save_fig4paper_plt
)

set_jax_device_config(cuda_device="0")

logger = get_logger()

def append_point(
    points_dict: dict[str, list], t_df: pd.DataFrame, g: float, xparam: float
):
    # 与えられたDataFrameからgとxparamに対応する列を抽出し, SxやMC,MLEなどの量とのセットでtuple点として追加する関数
    # xparamはpやsigmaなどのパラメータ, gはcoupling_strength
    # 全部リストの参照渡し

    # points_MCtrain, raw_points_MCtrain = (
    #     points_dict["points_MCtrain"],
    #     points_dict["raw_points_MCtrain"],
    # )
    # points_MCtest, raw_points_MCtest, _ = (
    #     points_dict["points_MCtest"],
    #     points_dict["raw_points_MCtest"],
    #     points_dict["ks"],
    # )

    tt_df = t_df.query("coupling_strength == @g")

    if (
        "critical_coupling_strength" not in tt_df.columns
        or tt_df["critical_coupling_strength"].isnull().all()
        or True
    ):
        # critical_coupling_strengthがない場合は, 1つのネットワークに対して1つのg_cしかないと仮定しているので, その値を計算する
        # もしくはg_cがNaNの場合は, そのネットワークに対してg_cが計算されていないとみなす
        logger.info(
            "critical_coupling_strength is not found in the DataFrame, calculating g_c..."
        )
        import pickle

        import jax

        from multid_rnn.dataclass.net_params import NetworkParameters
        from multid_rnn.net.net import critical_g_hetero2var

        rndkey = jax.random.PRNGKey(0)
        # assert (tt_df["pkl_path"] == tt_df["pkl_path"].iloc[0]).all()
        with open(tt_df["pkl_path"].iloc[0], "rb") as f:
            net_params: NetworkParameters = pickle.load(f)
            gc_jarray, _, rndkey = critical_g_hetero2var(
                net_params=net_params, rndkey=rndkey, n_samples=100000
            )
        gc = gc_jarray.item()
    else:
        gc = pd.unique(
            tt_df["critical_coupling_strength"]
        ).mean()  # 同一ネットワークを仮定しているためg_cは絶対1つだけど, サンプル平均でg_cを計算している場合は計算誤差があるかもしれないので一応平均を取る.
    is_g_ge_gc = g >= gc
    gFactor = g / gc

    #############
    # Sx系の処理 #
    #############
    if "df_SxSphi" in tt_df.columns and tt_df["df_SxSphi"].iloc[0] is not None:
        points_Sx, raw_points_Sx = (
            points_dict["points_Sx"],
            points_dict["raw_points_Sx"],
        )
        trialdfs = tt_df["df_SxSphi"]  # Sx系はdf_SxSphiに入っている

        # freqRange = trialdfs.iloc[0]["freqRange"]
        list_Sx = np.asarray([trialdfs.iloc[i]["Sx"] for i in range(trialdfs.shape[0])])
        meanedSx = np.mean(
            [trialdfs.iloc[i]["Sx"] for i in range(trialdfs.shape[0])], axis=0
        )
        medianedSx = np.median(
            [trialdfs.iloc[i]["Sx"] for i in range(trialdfs.shape[0])], axis=0
        )

        point_colum = [
            xparam,
            g,
            np.max(meanedSx),
            np.max(medianedSx),
            gc,
            is_g_ge_gc,
            gFactor,
        ]  # [xaxis, g, maxmeanSx, maxmedianSx, gc, is_g_ge_gc, gFactor]
        points_Sx.append(point_colum)

        max_f_Sxs = np.max(
            list_Sx, axis=1
        )  # trialsで平均化しないG(f)の最大値, trialの数だけ
        mean_max_f_Sxs = np.mean(max_f_Sxs)
        std_max_f_Sxs = np.std(max_f_Sxs)

        for max_f_Sx in max_f_Sxs:
            raw_points_Sx.append(
                [xparam, g, gc, is_g_ge_gc, max_f_Sx, mean_max_f_Sxs, std_max_f_Sxs]
            )

    # region MC系の処理
    #############
    # MC系の処理 #
    #############
    if "df_MC" in tt_df.columns and tt_df["df_MC"].iloc[0] is not None:
        dfs_MC = tt_df["df_MC"]  # trialをindexとして, dfが入っている
        tags = pd.unique(
            dfs_MC.iloc[0]["tag"]
        )  # 読み出しニューロン数やMCの計算の仕方など
        for tag in tags:
            rawkey = "points_totalMC_raw_" + tag
            if rawkey not in points_dict.keys():
                points_dict[rawkey] = []
            raw_points_MC = points_dict[rawkey]

            # 一旦はtotalMCのみをプロットする
            list_MC = np.asarray(
                [
                    pd.unique(
                        dfs_MC.iloc[i][dfs_MC.iloc[i]["tag"] == tag]["totalMC"]
                    ).item()
                    for i in range(dfs_MC.shape[0])
                ]
            )
            for mc in list_MC:
                raw_points_MC.append(
                    [xparam, g, gc, is_g_ge_gc, mc, np.mean(list_MC), np.std(list_MC)]
                )
    # endregion

    # region lyap系の処理
    ###############
    # lyap系の処理 #
    ###############
    if "df_lyap" in tt_df.columns and tt_df["df_lyap"].iloc[0] is not None:
        trialdfs = tt_df["df_lyap"]
        if "maxlyap" in trialdfs.iloc[0].columns:
            points_lyap, raw_points_lyap = (
                points_dict["points_lyap"],
                points_dict["raw_points_lyap"],
            )
            if trialdfs.iloc[0]["maxlyap"][0] is not None:
                list_lyap = np.asarray(
                    [trialdfs.iloc[i]["maxlyap"][0] for i in range(trialdfs.shape[0])]
                )

                point_colum = [
                    xparam,
                    g,
                    np.mean(list_lyap),
                    np.median(list_lyap),
                    gc,
                    is_g_ge_gc,
                    gFactor,
                ]
                points_lyap.append(point_colum)

                for lyap in list_lyap:
                    raw_points_lyap.append([xparam, g, gc, is_g_ge_gc, lyap])
    # endregion


def main(
    sim_dir_path: Path,
    infodict: dict,
    filter_pass: dict | None = None,
    filter_reject: dict | None = None,
    coeff_filter: dict | None = None,
    dir_prename="",
    dir_postname="",
    list_read_dataname: list[SAVED_DATA_TYPES] = ["SxSphix", "heterovars"],
):
    logger.info(f"load data from {sim_dir_path}")
    df_sims = load_net(
        sim_dir_path,
        filter_pass=filter_pass,
        filter_reject=filter_reject,
        coeff_filter=coeff_filter,
        recursive_df=True,
        list_read_dataname=list_read_dataname,
    )
    if df_sims.size == 0:
        raise Exception("no data found")
    logger.info(df_sims.info())
    logger.info(df_sims.head(5))

    hetero_var_label_key = "hetero_info_label"
    hetero_dist_prefix = "hetero_info_dist"
    additional_gc_line = None  # only for normal dist case

    # region 保存用Directoryの作成, 絞り込み条件の保存
    dump_path = (
        project_root
        / "results"
        / "dump"
        / "plot_data"
        / (
            dir_prename
            + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            + dir_postname
        )
    )
    dump_path.mkdir(exist_ok=True, parents=True)

    # df_sims内でuniqueな情報を保存 (あとで確認する用)
    if filter_pass is not None:
        infodict["filter_pass"] = filter_pass  # 絞り込み条件を追加
    if filter_reject is not None:
        infodict["filter_reject"] = filter_reject  # 絞り込み条件を追加

    # network_paramsの条件の抽出
    info_keys = list(df_sims.columns)

    def exists_remove(li: list, key: str):
        if key in li:
            li.remove(key)

    exists_remove(info_keys, "df_SxSphi")
    exists_remove(info_keys, "df_hetero")
    exists_remove(info_keys, "df_xtrace")
    exists_remove(info_keys, "df_MC")
    exists_remove(info_keys, "pkl_path")

    for key in info_keys:
        if df_sims[key].dtype == object:
            # object型の列は一旦文字列として扱ってuniqueを取ってから該当箇所を抽出する
            # もしくは flattenしてtuple df_sims[key].apply(lambda x: x.flatten()).apply(tuple).unique()
            us_str = df_sims[key].apply(str).unique()
            ret = []
            for u in us_str:
                for v in df_sims[key]:
                    if u == str(v):
                        ret.append(v)
                        break
            # ret = [np.asarray(v) for v in ]
        elif isinstance(df_sims[key].unique(), pd.arrays.StringArray):
            ret = df_sims[key].unique().astype(str).tolist()
        else:
            ret = df_sims[key].unique()
        infodict[key] = ret

    assert len(infodict["hetero_info_label"]) == 1
    heterovarname = infodict["hetero_info_label"][0]

    # 保存
    with open(os.path.join(dump_path, "info.json"), "w") as fp:
        json.dump(infodict, fp, indent=4, cls=NumpyJSONEncoder)
    # endregion

    # region dataframe から pointを抽出してプロット用のpointリストを作成
    """
    以下では読み込んだ入れ子Dataframeを操作し, 自分がプロットしたいものを列としたDataframeにする
    各種トライアル軸の平均などはappend_point関数内で行われる
    """

    points_dict = {
        "points_Sx": [],
        "raw_points_Sx": [],
        "points_lyap": [],
        "raw_points_lyap": [],
    }
    print("choose {}".format(target_dist_type))
    if target_dist_type == "TwoValDist":
        low_val_str = f"{hetero_dist_prefix}_TwoValDist_low_val"
        high_val_str = f"{hetero_dist_prefix}_TwoValDist_high_val"
        p_str = f"{hetero_dist_prefix}_TwoValDist_p"
        list_val1 = df_sims[low_val_str].unique()
        list_val2 = df_sims[high_val_str].unique()
        list_p = df_sims[p_str].unique()

        assert sum([v.size > 1 for v in [list_val1, list_val2, list_p]]) == 1, (
            "numebr of variable parameter of prob distribution must be 1"
        )

        if list_p.size > 1:
            # 以降はval1, val2固定でpを変化させた場合限定!!!
            assert list_val1.size == 1 and list_val2.size == 1, (
                "別のsimulationが混ざっているかも"
            )

            val1 = list_val1[0]
            val2 = list_val2[0]
            # print(f"val1={list_val1}, val2={list_val2}, p={list_p}")

            # あるpとあるgでdfをフィルタする（フィルタしたdfには1つのネットワーク設定の複数トライアルが含まれていることを仮定）
            for p in tqdm.tqdm(list_p, desc="quary data: p", leave=False):
                t_df = df_sims.query(
                    f"{high_val_str} == @val2 & {low_val_str} == @val1 & {p_str} == @p"
                )
                [
                    append_point(points_dict, t_df, g, xparam=p)
                    for g in t_df["coupling_strength"].unique()
                ]
            xparam_name = f"p (val1={val1},val2={val2})"  # gamma_dist_pだと味気無いのでわかりやすくval1とval2の情報も付与
        else:
            raise NotImplementedError(
                f"not supported yet, {[list_val1, list_val2, list_p]}"
            )
    elif target_dist_type == "UniformDist":
        min_val_str = f"{hetero_dist_prefix}_UniformDist_min_val"
        max_val_str = f"{hetero_dist_prefix}_UniformDist_max_val"
        list_min_val = np.asarray(pd.unique(df_sims[min_val_str]))
        list_max_val = np.asarray(pd.unique(df_sims[max_val_str]))
        list_length = np.asarray(pd.unique(df_sims[max_val_str] - df_sims[min_val_str]))
        list_center = np.asarray(
            pd.unique((df_sims[max_val_str] + df_sims[min_val_str]) / 2.0)
        )

        if list_center.size == 1:
            # 中心固定で長さを変える
            center = list_center.item()
            for li in tqdm.tqdm(list_length, desc="quary data: length", leave=False):
                min_val = center - li / 2.0  # noqa: F841
                max_val = center + li / 2.0  # noqa: F841
                t_df = df_sims.query(
                    f"{min_val_str} == @min_val & {max_val_str} == @max_val"
                )
                [
                    append_point(points_dict, t_df, g, xparam=li)
                    for g in t_df["coupling_strength"].unique()
                ]
                gs = t_df["coupling_strength"].unique()
            xparam_name = f"dist length (center={center})"
        elif list_center.size > 1 and (
            list_min_val.size == 1 or list_max_val.size == 1
        ):
            # 左右のどちらか固定で長さを変える
            if list_min_val.size == 1:
                min_val = list_min_val.item()
                for max_val in list_max_val:
                    t_df = df_sims.query(
                        f"{min_val_str} == @min_val & {max_val_str} == @max_val"
                    )
                    [
                        append_point(points_dict, t_df, g, xparam=max_val)
                        for g in t_df["coupling_strength"].unique()
                    ]
                    gs = t_df["coupling_strength"].unique()
                xparam_name = f"Uniform max_val (min_val={min_val})"
            elif list_max_val.size == 1:
                max_val = list_max_val.item()
                for min_val in list_min_val:
                    t_df = df_sims.query(
                        f"{min_val_str} == @min_val & {max_val_str} == @max_val"
                    )
                    [
                        append_point(points_dict, t_df, g, xparam=min_val)
                        for g in t_df["coupling_strength"].unique()
                    ]
                    gs = t_df["coupling_strength"].unique()
                xparam_name = f"Uniform min_val (max_val={max_val})"
            else:
                raise NotImplementedError(
                    f"not supported yet, {[list_min_val, list_max_val, list_length, list_center]}"
                )
        elif list_length.size == 1 and list_center.size > 1:
            # 長さ固定で中心を変える
            raise NotImplementedError()
        else:
            raise NotImplementedError(
                f"not supported yet, {[list_min_val, list_max_val, list_length, list_center]}"
            )
    elif target_dist_type == "TruncatedNormalDist":
        mean_val_str = f"{hetero_dist_prefix}_TruncatedNormalDist_mean"
        sigma_str = f"{hetero_dist_prefix}_TruncatedNormalDist_sigma"
        min_str = f"{hetero_dist_prefix}_TruncatedNormalDist_min"
        max_str = f"{hetero_dist_prefix}_TruncatedNormalDist_max"
        list_mean = np.asarray(pd.unique(df_sims[mean_val_str]))
        list_sigma = np.asarray(pd.unique(df_sims[sigma_str]))
        list_min = np.asarray(pd.unique(df_sims[min_str]))
        list_max = np.asarray(pd.unique(df_sims[max_str]))

        assert list_min.size == 1 and list_max.size == 1, (
            "min and max must be fixed, otherwise not supported yet"
        )
        if list_mean.size == 1 and list_sigma.size > 1:
            # 平均固定でsigmaを変える
            mean = list_mean.item()
            for sigma in tqdm.tqdm(list_sigma, desc="quary data: sigma", leave=False):
                t_df = df_sims.query(f"{mean_val_str} == @mean & {sigma_str} == @sigma")
                [
                    append_point(points_dict, t_df, g, xparam=sigma)
                    for g in t_df["coupling_strength"].unique()
                ]
            xparam_name = (
                f"sigma (mean={mean}), min={list_min.item()}, max={list_max.item()}"
            )
        elif list_mean.size > 1 and list_sigma.size == 1:
            # sigma固定で平均を変える
            raise NotImplementedError("sigma固定で平均を変えるのはまだ対応していない")
        else:
            raise NotImplementedError(
                f"not supported yet, {[list_mean, list_sigma, list_min, list_max]}"
            )
    elif target_dist_type == "normal":
        df_sims = df_sims.query(f'{hetero_dist_prefix}_dist_type == "normal"')
        list_mu = np.asarray(pd.unique(df_sims[f"{hetero_dist_prefix}_mu"]))
        assert list_mu.size == 1  # ここで引っかかるようならフィルタしてね〜
        list_sigma = np.asarray(pd.unique(df_sims[f"{hetero_dist_prefix}_sigma"]))
        list_sigma.sort()
        mu = list_mu[0]

        # あるpとあるgでdfをフィルタする（フィルタしたdfには1つのネットワーク設定の複数トライアルが含まれていることを仮定）
        for sigma in tqdm.tqdm(list_sigma, desc="quary data: sigma", leave=False):
            t_df = df_sims.query(
                f"{hetero_dist_prefix}_mu == @mu & {hetero_dist_prefix}_sigma == @sigma"
            )
            gs = pd.unique(t_df["g"])
            for g in tqdm.tqdm(gs, desc="quary data: g", leave=False):
                append_point(points_dict, t_df, g, sigma)
        xparam_name = f"sigma (mu={mu})"

        # empirical g_cの計算
        # dist = {
        # "dist_type": "normal",
        # "mu": mu,
        # }
        # freqRange = np.linspace(0, np.pi, 1000)
        # kwargs = {}
        # kwargs["D"] = 2
        # kwargs["net_type"] = ""
        # kwargs["gamma"] = df["gamma"].unique()[0]
        # from net.net import sample_rvs_iid, pure2varGf
        # list_gc_empavgGf = []
        # Nsample = 50000
        # for s in list_sigma:
        #     dist["sigma"] = s
        #     hetero_betas, _ = sample_rvs_iid(N, dist)
        #     empavgGvals = np.asarray([pure2varGf(kwargs['gamma'], beta, freqRange) for beta in hetero_betas]).mean(axis=0)
        #     list_gc_empavgGf.append(np.sqrt(1/empavgGvals[np.argmax(empavgGvals)]))
        # list_gc_empavgGf = np.asarray(list_gc_empavgGf)

        raise NotImplementedError(
            "まだ新しいnet.net.critical_g_mdheteroに対応していない"
        )
        # from net.net import critical_g_mdhetero

        # additional_gc_line = []
        # dist = {
        #     "dist_type": "normal",
        #     "mu": mu,
        # }
        # # hetero_var_name = np.unique([l[len("hetero_info_"):].split("_")[0] for l in list(df.columns) if "hetero_info" in l])[0]
        # net_params = {
        #     "net_type": df_sims["net_type"].unique()[0],
        #     "N": df_sims["N"].unique()[0],
        #     "D": df_sims["D"].unique()[0],
        #     "hetero_info": {hetero_var_name: dist},
        #     "gamma": df_sims["gamma"].unique()[0]
        #     if "gamma" in df_sims.columns
        #     else None,
        #     "beta": df_sims["beta"].unique()[0] if "beta" in df_sims.columns else None,
        #     "dmft_type": "1body",
        # }
        # for s in list_sigma:
        #     dist["sigma"] = s
        #     g_c1body = critical_g_mdhetero(**net_params)
        #     additional_gc_line.append(g_c1body)
        # additional_gc_line = np.asarray(additional_gc_line)
    else:
        raise Exception()

    points_dict["points_Sx"] = np.asarray(points_dict["points_Sx"])  # type: ignore

    # endregion

    # analytical_gc = np.asarray([critical_g(muGamma, sg, 0, mode="heteroGPA1var") for sg in sigmaGammaRange])
    # ax.plot(sigmaGammaRange, analytical_gc, zs=0, zdir='z', label=r'$g_c$', color=cm_heat.colors[0])

    font_size = 15
    mean_marker_size = 2
    each_marker_size = 7

    # region Sx系のプロット
    if points_dict["points_Sx"].size != 0:
        # region Sx(f)をtrial集合でmedian, mean操作をしたものの最大値 \max_f(\sum_{N_trial}{i=1}(S_{x,i}(f))) と \max_f(median({S_{x,i}(f)}_i={1,\dots,N_trial})) の3d, 2dプロット
        vis_df = pd.DataFrame(
            points_dict["points_Sx"][:, :6],
            columns=[xparam_name, "g", "max meanedSx", "max medianedSx", "gc", "g>=gc"],
        )
        vis_df["symbol"] = "g>=gc"
        vis_df["symbol"] = vis_df["symbol"].where(vis_df["g>=gc"] == 1, "g<gc")
        #
        # trialsでmean平均したSxの最大値をプロット
        #
        fig = px.scatter_3d(
            vis_df,
            x=xparam_name,
            y="g",
            z="max meanedSx",
            symbol="g>=gc",
            color="max meanedSx",
        )
        # グラフ全体とホバーのフォントサイズ変更
        fig.update_layout(font_size=font_size, hoverlabel_font_size=font_size)
        # カラーバーの長さを0.5に変更
        fig.update_layout(coloraxis=dict(colorbar=dict(len=0.5)))
        # 余白を削除する
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
        save_name = "max_meanedSx"
        save_mofify_fig(fig, os.path.join(dump_path, f"{save_name}.html"))

        plot3d_surface(
            vis_df,
            xparam_name,
            "g",
            "max meanedSx",
            "max_meanedSx_surface",
            dump_path,
            additional_gc_line=additional_gc_line,
        )
        plot2d_heatmap(
            vis_df,
            xparam_name,
            "g",
            "max meanedSx",
            "max_meanedSx_heatmap",
            dump_path,
            additional_gc_line=additional_gc_line,
        )
        plot2d_heatmap(
            vis_df,
            xparam_name,
            "g",
            "max meanedSx",
            "max_meanedSx_heatmap_cmax",
            dump_path,
            cmaxp=90,
            additional_gc_line=additional_gc_line,
        )

        #
        # trialsでmedian平均したSxの最大値をプロット
        #
        fig = px.scatter_3d(
            vis_df,
            x=xparam_name,
            y="g",
            z="max medianedSx",
            symbol="g>=gc",
            color="max medianedSx",
        )
        # グラフ全体とホバーのフォントサイズ変更
        fig.update_layout(font_size=font_size, hoverlabel_font_size=font_size)
        fig.update_layout(yaxis=dict(tickfont=dict(size=5)))
        # カラーバーの長さを0.5に変更
        fig.update_layout(coloraxis=dict(colorbar=dict(len=0.5)))
        # 余白を削除する
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
        save_name = "max_medianedSx"
        save_mofify_fig(fig, os.path.join(dump_path, f"{save_name}.html"))

        plot3d_surface(
            vis_df,
            xparam_name,
            "g",
            "max medianedSx",
            "max_medianedSx_surface",
            dump_path,
            additional_gc_line=additional_gc_line,
        )
        plot2d_heatmap(
            vis_df,
            xparam_name,
            "g",
            "max medianedSx",
            "max_medianedSx_heatmap",
            dump_path,
            additional_gc_line=additional_gc_line,
        )
        plot2d_heatmap(
            vis_df,
            xparam_name,
            "g",
            "max medianedSx",
            "max_medianedSx_heatmap_cmax",
            dump_path,
            cmaxp=90,
            additional_gc_line=additional_gc_line,
        )
        # plot3d_mesh(vis_df, xparam_name, 'g', 'max medianedSx', 'max_medianedSx_mesh')
        # endregion

        # region あるpにおいて, gfactorとmax Sxの関係の2dプロット用データ保存(論文用)
        xarray = np.asarray(pd.unique(vis_df[xparam_name]))
        xarray = xarray[np.argsort(xarray)]
        for xval in xarray:
            gs = vis_df[vis_df[xparam_name] == xval]["g"].to_numpy()
            argsortindex = np.argsort(gs)
            gs = gs[argsortindex]
            gcs = vis_df[vis_df[xparam_name] == xval]["gc"].to_numpy()[argsortindex]
            maxmeanedSx = vis_df[vis_df[xparam_name] == xval][
                "max meanedSx"
            ].to_numpy()[argsortindex]
            maxmedianedSx = vis_df[vis_df[xparam_name] == xval][
                "max medianedSx"
            ].to_numpy()[argsortindex]
            gfactors = gs / np.mean(gcs)
            np.savez(
                os.path.join(dump_path, f"maxaveragedSx_{xparam_name}={xval:.1f}.npz"),
                maxmeanedSx=maxmeanedSx,
                maxmedianedSx=maxmedianedSx,
                gfactors=gfactors,
                gs=gs,
                varname=xval,
                N=np.unique(infodict["n_neuron"])[0],
            )
        # endregion

        # region 各トライアルにおけるSxの最大値, つまり {max_f S_{x,i}(f)}_{i={1,\dots,N_trial}} をプロット

        #
        # trialsで平均化する前のSxの最大値をプロット
        #
        vis_df = pd.DataFrame(
            points_dict["raw_points_Sx"],
            columns=[
                xparam_name,
                "g",
                "gc",
                "g>=gc",
                "max Sx",
                "mean max Sx",
                "std max Sx",
            ],
        )
        vis_df["symbol"] = "g>=gc"
        vis_df["symbol"] = vis_df["symbol"].where(vis_df["g>=gc"] == 1, "g<gc")

        fig1 = px.scatter_3d(
            vis_df, x=xparam_name, y="g", z="max Sx", symbol="g>=gc", color="max Sx"
        )
        fig1.update_traces(marker_size=mean_marker_size)

        # グラフ全体とホバーのフォントサイズ変更
        fig1.update_layout(font_size=font_size, hoverlabel_font_size=font_size)
        # カラーバーの長さを0.5に変更
        fig1.update_layout(coloraxis=dict(colorbar=dict(len=0.5)))
        # 余白を削除する
        fig1.update_layout(margin=dict(l=0, r=0, b=0, t=0))
        save_name = "max_Sx"
        save_mofify_fig(fig1, os.path.join(dump_path, f"{save_name}.html"))

        #
        # trialsで平均化する前のSxの最大値の平均と分散をプロット （なんかプロットできてない）
        #
        fig2 = px.scatter_3d(
            vis_df,
            x=xparam_name,
            y="g",
            z="mean max Sx",
            error_z="std max Sx",
            symbol="g>=gc",
            color="mean max Sx",
        )
        fig2.update_traces(error_z=dict(color="black"))
        fig2.update_traces(marker_size=each_marker_size)

        save_name = "max_Sx_witherror"
        save_mofify_fig(fig2, os.path.join(dump_path, f"{save_name}.html"))

        # list_p = df[f"{hetero_dist_prefix}_p"].unique()
        # list_p.sort()
        # list_g_c = []
        # list_z = []
        # for p in list_p:
        #     t_g_c = df[df[f"{hetero_dist_prefix}_p"]==p]["g_c"].unique()[0]
        #     list_g_c.append(t_g_c)
        #     list_z.append(0)
        # g_c_line_df = pd.DataFrame.from_dict({xparam_name: list_p, "g": list_g_c, "max Sx": list_z})
        # fig3 = px.line_3d(g_c_line_df, x=xparam_name, y="g", z="max Sx")
        # fig3.update_traces(
        #     # marker=dict(
        #     #     color='lightblue',
        #     #     size=10),
        #     line=dict(
        #         color='green',
        #         width=12
        #     )
        # )

        # region 転移点のプロット
        fig3 = plot_gc_line(df_sims, hetero_var_label_key)
        if target_dist_type == "normal":
            """
            ヘテロ性が正規分布に従うとき, 先行研究の1bodyで計算したg_cをプロットする.
            """
            fig4 = plot_gc_line(
                df_sims,
                hetero_var_label_key,
                list_g_c=additional_gc_line,
                linecolor="white",
                dash="dash",
            )
            fig3 = go.Figure(data=fig3.data + fig4.data, layout=fig3.layout)  # type: ignore
        # pio.write_html(fig3, os.path.join(dump_path, "g_c_line.html"))

        all_fig = go.Figure(data=fig1.data + fig2.data + fig3.data, layout=fig1.layout)  # type: ignore
        save_mofify_fig(all_fig, os.path.join(dump_path, "max_Sx_all.html"))

        # endregion

        # endregion
    # endregion

    # region リアプノフ指数のプロット
    if len(points_dict["points_lyap"]) != 0:
        vis_df_each = pd.DataFrame(
            points_dict["raw_points_lyap"],
            columns=[xparam_name, "g", "gc", "g>=gc", "maximum lyap", "seed"],
        )
        vis_df_each["symbol"] = "g>=gc"
        vis_df_each["symbol"].where(vis_df_each["g>=gc"] == 1, "g<gc", inplace=True)

        fig1 = px.scatter_3d(
            vis_df_each,
            x=xparam_name,
            y="g",
            z="maximum lyap",
            symbol="g>=gc",
            color="maximum lyap",
        )
        fig1.update_traces(marker_size=mean_marker_size)

        # グラフ全体とホバーのフォントサイズ変更
        fig1.update_layout(font_size=font_size, hoverlabel_font_size=font_size)
        # カラーバーの長さを0.5に変更
        fig1.update_layout(coloraxis=dict(colorbar=dict(len=0.5)))
        # 余白を削除する
        fig1.update_layout(margin=dict(l=0, r=0, b=0, t=0))
        save_mofify_fig(fig1, os.path.join(dump_path, "lyap_each.html"))

        vis_df = pd.DataFrame(
            points_dict["points_lyap"],
            columns=[
                xparam_name,
                "g",
                "mean maximum lyap",
                "median maximum lyap",
                "gc",
                "g>=gc",
                "gFactor",
            ],
        )
        fig2 = px.scatter_3d(
            vis_df,
            x=xparam_name,
            y="g",
            z="mean maximum lyap",
            symbol="g>=gc",
            color="mean maximum lyap",
        )
        fig2.update_traces(marker_size=each_marker_size)

        plot3d_surface(
            vis_df,
            xparam_name,
            "g",
            "mean maximum lyap",
            "lyap_mean_surface",
            dump_path,
            is0surface=True,
        )
        save_mofify_fig(fig2, os.path.join(dump_path, "lyap_mean.html"))

        # list_p = df[f"{hetero_dist_prefix}_p"].unique()
        # list_p.sort()
        # list_g_c = []
        # list_z = []
        # for p in list_p:
        #     t_g_c = df[df[f"{hetero_dist_prefix}_p"]==p]["g_c"].unique()[0]
        #     list_g_c.append(t_g_c)
        #     list_z.append(0)
        # g_c_line_df = pd.DataFrame.from_dict({xparam_name: list_p, "g": list_g_c, "maximum lyap": list_z})
        # fig3 = px.line_3d(g_c_line_df, x=xparam_name, y="g", z="maximum lyap")
        # fig3.update_traces(
        #     # marker=dict(
        #     #     color='lightblue',
        #     #     size=10),
        #     line=dict(
        #         color='green',
        #         width=12
        #     )
        # )
        fig3 = plot_gc_line(df_sims, hetero_var_label_key)

        xarray = np.sort(np.asarray(pd.unique(vis_df[xparam_name])))
        yarray = np.sort(np.asarray(pd.unique(vis_df["g"])))
        zmatrix = np.zeros([yarray.size, xarray.size])

        surf = go.Surface(
            x=xarray,
            y=yarray,
            z=zmatrix,
            opacity=0.7,
            showscale=False,
            cmin=0,
            cmax=1,  # カラーバーの1番下の色にする
            #   cmin=np.min(vis_df['mean maximum lyap']),cmax=np.max(vis_df['mean maximum lyap'])
        )
        fig4 = go.Figure(data=[surf])

        all_fig = go.Figure(
            data=fig1.data + fig2.data + fig3.data + fig4.data,  # type: ignore
            layout=fig1.layout,  # type: ignore
        )
        save_mofify_fig(all_fig, os.path.join(dump_path, "lyap_all.html"))

        """
        あるpやNにおいて, gfactorとlyapunov指数の関係をプロットする用のファイル保存(論文用)
        """
        for xval in xarray:
            gs = vis_df[vis_df[xparam_name] == xval]["g"].to_numpy()
            argsortindex = np.argsort(gs)
            gs = gs[argsortindex]
            gcs = vis_df[vis_df[xparam_name] == xval]["gc"].to_numpy()[argsortindex]
            meanmaxlyaps = vis_df[vis_df[xparam_name] == xval][
                "mean maximum lyap"
            ].to_numpy()[argsortindex]
            medmaxlyaps = vis_df[vis_df[xparam_name] == xval][
                "median maximum lyap"
            ].to_numpy()[argsortindex]
            gfactors = gs / np.mean(gcs)

            np.savez(
                os.path.join(dump_path, f"lyap_gfactor_{xparam_name}={xval}.npz"),
                meanmaxlyaps=meanmaxlyaps,
                medmaxlyaps=medmaxlyaps,
                gfactors=gfactors,
                gs=gs,
                varname=xval,
                N=np.unique(infodict["n_neuron"])[0],
            )

            gs_each = vis_df_each[vis_df_each[xparam_name] == xval]["g"].to_numpy()
            gcs_each = vis_df_each[vis_df_each[xparam_name] == xval]["gc"].to_numpy()
            maxlyaps = vis_df_each[vis_df_each[xparam_name] == xval][
                "maximum lyap"
            ].to_numpy()
            seed_each = vis_df_each[vis_df_each[xparam_name] == xval]["seed"].to_numpy()
            # argsortindex_each = np.argsort(gs_each)
            # gs_each = gs_each[argsortindex_each]
            # gcs_each = vis_df_each[vis_df_each[xparam_name] == xval]["gc"].to_numpy()[argsortindex_each]
            # maxlyaps = vis_df_each[vis_df_each[xparam_name] == xval]["maximum lyap"].to_numpy()[argsortindex_each]
            # seed_each = vis_df_each[vis_df_each[xparam_name] == xval]["seed"].to_numpy()[argsortindex_each]
            gfactors_each = gs_each / np.mean(gcs_each)
            np.savez(
                os.path.join(dump_path, f"lyap_each_gfactor_{xparam_name}={xval}.npz"),
                varname=xval,
                N=np.unique(infodict["n_neuron"])[0],
                seed_each=seed_each,
                maxlyaps=maxlyaps,
                gfactors_each=gfactors_each,
                gs_each=gs_each,
            )

    # endregion

    # region MCのプロット

    points_totalMC_keys = [
        key for key in points_dict.keys() if key.startswith("points_totalMC_")
    ]
    for key in points_totalMC_keys:
        if len(points_dict[key]) != 0:
            vis_df = pd.DataFrame(
                points_dict[key],
                columns=[
                    xparam_name,
                    "g",
                    "gc",
                    "g>=gc",
                    "MC",
                    "mean MC",
                    "std MC",
                ],
            )

            def plot_MC_3d(vis_df, f_name):
                #
                # trialsで平均化する前のMCをプロット
                #
                vis_df["symbol"] = "g>=gc"
                vis_df["symbol"] = vis_df["symbol"].where(vis_df["g>=gc"] == 1, "g<gc")

                fig1 = px.scatter_3d(
                    vis_df, x=xparam_name, y="g", z="MC", symbol="g>=gc", color="MC"
                )
                fig1.update_traces(marker_size=2)

                # グラフ全体とホバーのフォントサイズ変更
                fig1.update_layout(font_size=20, hoverlabel_font_size=20)
                # カラーバーの長さを0.5に変更
                fig1.update_layout(coloraxis=dict(colorbar=dict(len=0.5)))
                # 余白を削除する
                fig1.update_layout(margin=dict(l=0, r=0, b=0, t=0))
                save_mofify_fig(
                    fig1, os.path.join(dump_path, f"{f_name}_eachpoint.html")
                )

                #
                # trialsで平均化する前のMCの平均と分散をプロット
                #
                fig2 = px.scatter_3d(
                    vis_df,
                    x=xparam_name,
                    y="g",
                    z="mean MC",
                    error_z="std MC",
                    symbol="g>=gc",
                    color="mean MC",
                )
                fig2.update_traces(marker_size=5, error_z=dict(color="black"))

                save_mofify_fig(
                    fig2, os.path.join(dump_path, f"{f_name}_witherror.html")
                )

                all_fig = go.Figure(data=fig1.data + fig2.data, layout=fig1.layout)  # type: ignore
                save_mofify_fig(
                    all_fig,
                    os.path.join(dump_path, f"{f_name}_eachpointswitherror.html"),
                )

            plot_MC_3d(
                vis_df,
                f_name=f"totalMC_{key}",
            )

            plot2d_heatmap(
                target_df=vis_df,
                target_x_name=xparam_name,
                target_y_name="g",
                target_z_name="MC",
                title=f"totalMC_{key}",
                dump_path=dump_path,
                averageing_method="mean",
            )

            plot3d_surface(
                target_df=vis_df,
                target_x_name=xparam_name,
                target_y_name="g",
                target_z_name="MC",
                title=f"totalMC_{key}_surface",
                dump_path=dump_path,
                is0surface=False,
                averageing_method="mean",
            )

            # 横軸ヘテロ分布のパラメタ, 縦軸MC, ラベルcoupling_strengthの2dプロット
            
            cmap = plt.get_cmap("viridis")
            _xparam_name = xparam_name.replace('_', ' ')
            if pd.unique(vis_df["g"]).size <= pd.unique(vis_df[xparam_name]).size:
                fig, ax = plt.subplots(1,1,figsize=[8,5])
                gs = np.sort(np.asarray(pd.unique(vis_df["g"])))
                colors = []
                interporate_flip_xparams = []
                for idx_g, g in enumerate(gs):
                    g_df = vis_df.query(f"g == {g}")
                    if g_df.empty:
                        continue
                    xparams = np.sort(np.asarray(pd.unique(g_df[xparam_name])))
                    flip_xparams = g_df[g_df["g>=gc"]][xparam_name].min()
                    interporate_flip_xparams.append((flip_xparams + xparams[np.where(xparams == flip_xparams)[0][0] -1])/2.)
                    meanMC = [g_df[g_df[xparam_name] == xparam]["MC"].mean().item() for xparam in xparams]
                    line, = ax.plot(xparams, meanMC, label=r"$g=$"+f"{g}", color=cmap(idx_g/(gs.size-1)))
                    colors.append(line.get_color())
                ylim = ax.get_ylim()
                for i_g, g in enumerate(gs):
                    ax.vlines(x=interporate_flip_xparams[i_g], ymin=ylim[0], ymax=ylim[1], linestyles="--", colors=colors[i_g])
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
                ax.set_ylabel("Memory Capacity")
                ax.set_xlabel(_xparam_name)
                save_fig4paper_plt(fig, dirpath=dump_path, filename=f"totalMC2d_{key}_gs={gs}")


            # 横軸coupling_strength, 縦軸MC, ラベルヘテロ分布のパラメタの2dプロット
            if pd.unique(vis_df["g"]).size >= pd.unique(vis_df[xparam_name]).size:
                fig, ax = plt.subplots(1,1,figsize=[10,5])
                xparams = np.sort(np.asarray(pd.unique(vis_df[xparam_name])))
                colors = []
                interporate_flip_gs = []
                for idx_xparam, xparam_value in enumerate(xparams):
                    xparam_df = vis_df[vis_df[xparam_name] == xparam_value]
                    if xparam_df.empty:
                        continue
                    gs = np.sort(np.asarray(pd.unique(xparam_df["g"])))
                    flip_gs = xparam_df[xparam_df["g>=gc"]]["g"].min()
                    interporate_flip_gs.append((flip_gs + gs[np.where(gs == flip_gs)[0][0] -1])/2.)
                    meanMC = [xparam_df[xparam_df["g"] == g]["MC"].mean().item() for g in gs]
                    line, = ax.plot(gs, meanMC, label=f"{_xparam_name}={xparam_value:.3f}", color=cmap(idx_xparam/(xparams.size-1)))
                    colors.append(line.get_color())
                ylim = ax.get_ylim()
                for i_xparam, xparam_value in enumerate(xparams):
                    ax.vlines(x=interporate_flip_gs[i_xparam], ymin=ylim[0], ymax=ylim[1], linestyles="--", colors=colors[i_xparam])
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
                ax.set_ylabel("Memory Capacity")
                ax.set_xlabel("Coupling Strength")
                save_fig4paper_plt(fig, dirpath=dump_path, filename=f"totalMC2d_{key}_{xparam_name}={xparams}")

            # line_width = 3
            # linecolors = px.colors.qualitative.Plotly
            # gs = np.asarray(pd.unique(vis_df["g"]))
            # fig_all = go.Figure(data=[])
            # for i_g, g in enumerate(gs):
            #     g_df = vis_df.query(f"g == {g}")
            #     if g_df.empty:
            #         continue
            #     xparams = np.sort(np.asarray(pd.unique(g_df[xparam_name])))
            #     flip_xparams = g_df[g_df["g>=gc"]][xparam_name].min()
            #     interporate_flip_xparams = (flip_xparams + xparams[np.where(xparams == flip_xparams)[0][0] -1])/2.
            #     meanMC = [g_df[g_df[xparam_name] == xparam]["MC"].mean().item() for xparam in xparams]
            #     fig = go.Figure(data=[go.Scatter(x=xparams, y=meanMC, mode="lines+markers", name=f"g={g}")])
            #     fig.update_traces(line=dict(color=linecolors[i_g], width=line_width))
            #     fig.add_vline(
            #         x=interporate_flip_xparams, # type: ignore
            #         line_width=line_width,
            #         line_dash="dash",
            #         line_color=linecolors[i_g],
            #         opacity=.5,
            #     )
            #     fig_all = concat_figdata(fig_all, fig) # type: ignore
            #     save_mofify_fig(fig, dump_path / f"totalMC2d_{key}_g={g}.html", is_2d=True)
            # save_mofify_fig(fig_all, dump_path / f"totalMC2d_{key}_gs={gs}.html", is_2d=True)

    # region ascendとdescendの比較
    points_ascend_totalMC_keys = [key for key in points_totalMC_keys if "ascend" in key]
    for ascend_key in points_ascend_totalMC_keys:
        ascend_df = pd.DataFrame(
            points_dict[ascend_key],
            columns=[
                xparam_name,
                "g",
                "gc",
                "g>=gc",
                "MC",
                "mean MC",
                "std MC",
            ],
        )
        desencd_key = ascend_key.replace("ascend", "descend")
        descend_df = pd.DataFrame(
            points_dict[desencd_key],
            columns=[
                xparam_name,
                "g",
                "gc",
                "g>=gc",
                "MC",
                "mean MC",
                "std MC",
            ],
        )
        vis_df = descend_df.copy()
        vis_df["mean MC"] = descend_df["mean MC"] - ascend_df["mean MC"]

        plot2d_heatmap(
            target_df=vis_df,
            target_x_name=xparam_name,
            target_y_name="g",
            target_z_name="mean MC",
            title=f"totalMC_desend-ascend_{ascend_key.replace('ascend', '')}",
            dump_path=dump_path,
            averageing_method="mean",
            is_zero_mid=True,
        )
    # endregion
    # endregion


if __name__ == "__main__":
    """
    概要
    解析的な転移点と数値実験のデータをまとめてプロットする

    1. 保存したnpzファイルを読み込んでDataframe化 
        TODO: Dataframeを条件で読み込まないようにする Done
    2. DataFrameを結合
    3. クエリで対象データを抽出(DataFrameが大きいとクソ遅い)
    4. プロット
    """

    project_root = Path(
        os.environ.get("PROJECT_DIR", Path(__file__).resolve().parents[2])
    )

    ################################################################################################################
    if False:
        infodict = {}
        sim_path = Path("results") / "test_run_multi_sim" / "GammaheteroGPA2var"
        N = 3000
        target_dist_type = "TwoValDist"

        filter_pass = {
            "n_neuron": 3000,
            "hetero_info_label": "gamma",
            f"hetero_info_dist_{target_dist_type}_high_val": 10.0,
            f"hetero_info_dist_{target_dist_type}_low_val": 1.0,
        }
        coeff_filter = {
            "index": [1, 0],  # beta
            "value": 0.5,
        }
        main(
            project_root / sim_path,
            infodict,
            filter_pass=filter_pass,
            coeff_filter=coeff_filter,
        )

    ################################################################################################################
    if False:
        infodict = {}
        sim_path = Path("results/figN_gamma2valADA2var")
        N = 3000
        target_dist_type = "TwoValDist"

        filter_pass = {
            "n_neuron": 3000,
            "hetero_info_label": "gamma",
            f"hetero_info_dist_{target_dist_type}_high_val": 4.0,
            f"hetero_info_dist_{target_dist_type}_low_val": 1.0,
        }
        coeff_filter = {
            "index": [1, 0],  # beta
            "value": -2.0,
        }
        main(
            project_root / sim_path,
            infodict,
            filter_pass=filter_pass,
            coeff_filter=coeff_filter,
        )

    ################################################################################################################
    # 論文S2 
    if False:
        infodict = {}
        sim_path = Path("results/figN2_gammaUniGPA2var")
        N = 3000
        target_dist_type = "UniformDist"

        filter_pass = {
            "n_neuron": 3000,
            "hetero_info_label": "gamma",
        }
        coeff_filter = {
            "index": [1, 0],  # beta
            "value": 2.0,
        }
        main(
            project_root / sim_path,
            infodict,
            filter_pass=filter_pass,
            coeff_filter=coeff_filter,
        )

    ################################################################################################################
    if False:
        infodict = {}
        sim_path = Path("results/figN3_gammaUniADA2var")
        N = 3000
        target_dist_type = "UniformDist"

        filter_pass = {
            "n_neuron": 3000,
            "hetero_info_label": "gamma",
        }
        coeff_filter = {
            "index": [1, 0],  # beta
            "value": -2.0,
        }
        main(
            project_root / sim_path,
            infodict,
            filter_pass=filter_pass,
            coeff_filter=coeff_filter,
        )

    ################################################################################################################
    # 論文S2 
    if True:
        infodict = {}
        sim_path = Path("results/figN4_betaTrNormalGPAADA2var")
        N = 3000
        target_dist_type = "TruncatedNormalDist"

        filter_pass = {
            "n_neuron": 3000,
            "hetero_info_label": "beta",
        }
        coeff_filter = {
            "index": [1, 1],  # gamma
            "value": -3.0,
        }
        main(
            project_root / sim_path,
            infodict,
            filter_pass=filter_pass,
            coeff_filter=coeff_filter,
        )

    ################################################################################################################
    if False:
        infodict = {}
        sim_path = Path("results/figN6_betaUniGPA2var_gaussMC")
        N = 3000
        target_dist_type = "UniformDist"

        filter_pass = {
            "n_neuron": 3000,
            "hetero_info_label": "beta",
        }
        # filter_reject = {
        #     "coupling_strength": [0.0, 3.0]  # coupling_strengthが0のものは除外
        # }
        filter_reject = {}
        coeff_filter = {
            "index": [1, 1],  # gamma
            "value": -3.0,
        }
        main(
            project_root / sim_path,
            infodict,
            filter_pass=filter_pass,
            filter_reject=filter_reject,
            coeff_filter=coeff_filter,
            list_read_dataname=["MC"],
        )

    ################################################################################################################

    if False:
        infodict = {}
        sim_path = Path("results/figN7_betaUniADA2var_gaussMC")
        N = 3000
        target_dist_type = "UniformDist"

        filter_pass = {
            "n_neuron": 3000,
            "hetero_info_label": "beta",
        }
        # filter_reject = {
        #     "coupling_strength": [0.0, 3.0]  # coupling_strengthが0のものは除外
        # }
        filter_reject = {}
        coeff_filter = {
            "index": [1, 1],  # gamma
            "value": -3.0,
        }
        main(
            project_root / sim_path,
            infodict,
            filter_pass=filter_pass,
            filter_reject=filter_reject,
            coeff_filter=coeff_filter,
            list_read_dataname=["MC"],
        )

    ################################################################################################################
    if False:
        infodict = {}
        sim_path = Path("results/figN8_betaUniGPAADA2var_gaussMC")
        N = 3000
        target_dist_type = "UniformDist"

        filter_pass = {
            "n_neuron": 3000,
            "hetero_info_label": "beta",
        }
        # filter_reject = {
        #     "coupling_strength": [0.0, 3.0]  # coupling_strengthが0のものは除外
        # }
        filter_reject = {}
        coeff_filter = {
            "index": [1, 1],  # gamma
            "value": -3.0,
        }
        main(
            project_root / sim_path,
            infodict,
            filter_pass=filter_pass,
            filter_reject=filter_reject,
            coeff_filter=coeff_filter,
            list_read_dataname=["MC"],
        )

    ################################################################################################################
    
    if False:
        infodict = {}
        sim_path = Path("results/figN9_betaUniGPA2var_gaussMC_roughg")
        N = 3000
        target_dist_type = "UniformDist"

        filter_pass = {
            "n_neuron": 3000,
            "hetero_info_label": "beta",
        }
        filter_reject = {
            "coupling_strength": [0.9, 0.75]  # coupling_strengthがのものは除外
        }
        coeff_filter = {
            "index": [1, 1],  # gamma
            "value": -3.0,
        }
        main(
            project_root / sim_path,
            infodict,
            filter_pass=filter_pass,
            filter_reject=filter_reject,
            coeff_filter=coeff_filter,
            list_read_dataname=["MC"],
        )

    ################################################################################################################
    # 論文S4_MC
    if False:
        infodict = {}
        sim_path = Path("results/figN10_betaUniGPA2var_gaussMC_roughwidth")
        N = 3000
        target_dist_type = "UniformDist"

        filter_pass = {
            "n_neuron": 3000,
            "hetero_info_label": "beta",
        }
        coeff_filter = {
            "index": [1, 1],  # gamma
            "value": -3.0,
        }
        main(
            project_root / sim_path,
            infodict,
            filter_pass=filter_pass,
            filter_reject=None,
            coeff_filter=coeff_filter,
            list_read_dataname=["MC"],
        )
