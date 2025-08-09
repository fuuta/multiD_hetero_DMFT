from datetime import datetime
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo

import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots  # noqa: F401
from plotly import express as px
from plotly import graph_objects as go
from plotly import io as pio

plt.style.use(["science", "no-latex"])

"""
plotly で write_image を使うためには, 
kaleido を uv でインストールしておく必要がある.
`uv add "kaleido==0.2.1" --force-reinstall`
"""


def save_fig4paper_plt(
    fig: matplotlib.figure.Figure,
    dirpath: Path,
    filename: str | None = None,
    dpi: int = 400,
    bbox_inches="tight",
    transparent: bool = False,
    remove_upright_tickline: bool = True,
    is_force_remove_grid=True,
    fig_scales=[0.5, 0.75, 1.5, 2.0, 1],
):
    base_figsize = fig.get_size_inches()
    if not dirpath.exists():
        dirpath.mkdir(parents=True, exist_ok=True)
    if filename is None:
        filepath = (
            dirpath
            / f"{datetime.now(ZoneInfo('Asia/Tokyo')).strftime('%Y%m%d_%H%M%S')}"
        )
    else:
        filepath = dirpath / filename
    ax: matplotlib.axes._axes.Axes
    for ax in fig.axes:
        if remove_upright_tickline:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(axis="x", which="both", top=False)
            ax.tick_params(axis="y", which="both", right=False)
        if is_force_remove_grid:
            ax.grid(False)
    for scale in fig_scales:
        figsize = base_figsize * scale
        fig.set_size_inches(figsize)
        fig.tight_layout()
        fig.savefig(
            str(filepath) + f"_fs[{figsize[0].item()}, {figsize[1].item()}].png",
            dpi=dpi,
            bbox_inches=bbox_inches,
            transparent=transparent,
            pad_inches=0.05,
        )


def save_mofify_fig(fig, path, font_size=20, is_2d=False, dpi=400):
    # グラフ全体とホバーのフォントサイズ変更
    fig.update_layout(font_size=font_size, hoverlabel_font_size=font_size)
    axis_prop = dict(tickfont=dict(size=font_size))
    fig.update_layout(scene=dict(xaxis=axis_prop, yaxis=axis_prop, zaxis=axis_prop))
    fig.update_layout(font_family="Times New Roman, Times, serif", font_color="black")
    pio.write_html(fig, path)
    if is_2d:
        pio.write_image(fig, str(path) + "300.png", width=300, height=300)
        pio.write_image(fig, str(path) + "400.png", width=400, height=400)
        pio.write_image(fig, str(path) + "500.png", width=500, height=500)
        pio.write_image(fig, str(path) + "750.png", width=750, height=750)
        pio.write_image(fig, str(path) + "1000.png", width=1000, height=1000)


def concat_figdata(
    fig1: go.Figure,
    fig2: go.Figure,
    layout: go.Layout | None = None,
    merge_layouts: bool = False,
) -> go.Figure:
    """
    2つのgo.Figureを結合し、shapes/annotationsも保持する

    Parameters:
    -----------
    fig1, fig2 : go.Figure
        結合するFigure
    layout : go.Layout | None
        使用するレイアウト。Noneの場合は新しいレイアウトを作成
    merge_layouts : bool
        Trueの場合、両方のFigureのレイアウト要素をマージ
    """
    assert isinstance(fig1.data, (list, tuple))
    assert isinstance(fig2.data, (list, tuple))

    # レイアウトの決定
    if layout is None:
        if merge_layouts:
            # 両方のレイアウトをマージ
            layout = go.Layout(
                title=fig1.layout.title or fig2.layout.title,
                xaxis=fig1.layout.xaxis
                if fig1.layout.xaxis.title
                else fig2.layout.xaxis,
                yaxis=fig1.layout.yaxis
                if fig1.layout.yaxis.title
                else fig2.layout.yaxis,
            )
        else:
            layout = go.Layout()  # 空のレイアウト

    new_fig = go.Figure(data=list(fig1.data) + list(fig2.data), layout=layout)

    # shapesの追加（重複チェック付き）
    added_shapes = set()
    for fig in [fig1, fig2]:
        if hasattr(fig, "layout") and fig.layout.shapes:
            for shape in fig.layout.shapes:
                shape_key = (shape.type, shape.x0, shape.y0, shape.x1, shape.y1)
                if shape_key not in added_shapes:
                    new_fig.add_shape(shape)
                    added_shapes.add(shape_key)

    # annotationsの追加（重複チェック付き）
    added_annotations = set()
    for fig in [fig1, fig2]:
        if hasattr(fig, "layout") and fig.layout.annotations:
            for annotation in fig.layout.annotations:
                ann_key = (annotation.text, annotation.x, annotation.y)
                if ann_key not in added_annotations:
                    new_fig.add_annotation(annotation)
                    added_annotations.add(ann_key)

    return new_fig


def plot_gc_line(
    df,
    heterovar_label_key,
    z_height: float | None = 0.0,
    linecolor="red",
    gcname="critical_coupling_strength",
    list_g_c=None,
    dash=None,
    width=3,
) -> go.Figure:
    list_var = df[heterovar_label_key].unique()
    t_list_g_c = []
    list_z = []
    if list_g_c is None:
        for var in list_var:
            # t_g_c = df[df[f"{hetero_dist_prefix}_{varname}"]==var][gcname].unique()[0]
            t_g_c = df[df[heterovar_label_key] == var][gcname].unique()[0]
            t_list_g_c.append(t_g_c)
            list_z.append(z_height)
        list_g_c = t_list_g_c
    else:
        list_z = [z_height for i in range(len(list_g_c))]

    g_c_line_df = pd.DataFrame.from_dict(
        {heterovar_label_key: list_var, "g": list_g_c, "max Sx": list_z}
    )
    if z_height is not None:
        fig = px.line_3d(g_c_line_df, x=heterovar_label_key, y="g", z="max Sx")
    else:
        fig = px.line(g_c_line_df, x=heterovar_label_key, y="g")
    linedict = dict(color=linecolor, width=width)
    if dash is not None:
        linedict["dash"] = dash
    fig.update_traces(
        # marker=dict(
        #     color='lightblue',
        #     size=10),
        line=linedict
    )
    return fig


def plot3d_surface(
    target_df,
    target_x_name,
    target_y_name,
    target_z_name,
    title,
    dump_path: Path,
    clipz=None,
    is0surface=False,
    contours_dict=None,
    averageing_method: Literal["mean", "median", None] = None,
    **kwargs,
):
    font_size = kwargs.get("font_size", 10)
    xarray = np.sort(pd.unique(target_df[target_x_name]))
    yarray = np.sort(pd.unique(target_df[target_y_name]))
    zmatrix = np.empty([yarray.size, xarray.size])
    for i_x, x in enumerate(xarray):
        for i_y, y in enumerate(yarray):
            temp_df = target_df[
                (target_df[target_x_name] == x) & (target_df[target_y_name] == y)
            ]
            if temp_df.index.size == 0:
                zmatrix[i_y][i_x] = None
            elif temp_df.index.size == 1:
                zmatrix[i_y][i_x] = temp_df[target_z_name].values[0]
            else:
                if averageing_method == "mean":
                    zmatrix[i_y][i_x] = temp_df[target_z_name].values.mean()
                elif averageing_method == "median":
                    zmatrix[i_y][i_x] = temp_df[target_z_name].values.median()
                elif averageing_method is None:
                    raise ValueError("specify averageing_method")
                else:
                    raise ValueError(
                        f"averageing_method {averageing_method} is not supported"
                    )

    zmin, zmax = np.min(target_df[target_z_name]), np.max(target_df[target_z_name])
    if is0surface:
        zmin, zmax = -max(np.abs(zmin), np.abs(zmax)), max(np.abs(zmin), np.abs(zmax))
    if clipz is not None:
        zmatrix = np.clip(zmatrix, clipz[0], clipz[1])
    if contours_dict is None:
        is_show_contours = False
        if is0surface:
            contours_dict = {
                "z": {"show": is_show_contours, "start": 0.0, "end": 0, "size": 0.05}
            }
        else:
            contours_dict = {
                "z": {"show": is_show_contours, "start": 0.0, "end": 1, "size": 0.05}
            }  # z値を基準に等高線を作成
        contours_dict = None
    surf = go.Surface(
        x=xarray,
        y=yarray,
        z=zmatrix,  # xとyはそれぞれmesh化する前の1darray，zは対応する2darrayになる
        contours=contours_dict,
        opacity=0.9,
        cmin=zmin,
        cmax=zmax,
        coloraxis="coloraxis",
    )

    if is0surface:
        surf0 = go.Surface(
            x=xarray,
            y=yarray,
            z=zmatrix * 0,
            opacity=0.7,
            showscale=False,
            cmin=0,
            cmax=1,
            surfacecolor=np.zeros_like(zmatrix),
            colorscale="hot",
        )
        surf0_opacity1 = go.Surface(
            x=xarray,
            y=yarray,
            z=zmatrix * 0,
            opacity=1.0,
            showscale=False,
            cmin=0,
            cmax=1,
            surfacecolor=np.zeros_like(zmatrix),
            colorscale="hot",
        )
        fig = go.Figure([surf])
        fig0 = go.Figure([surf0])
        fig0_opacity1 = go.Figure([surf0_opacity1])
    else:
        fig = go.Figure([surf])
        fig0 = go.Figure([])
        fig0_opacity1 = go.Figure([])

    # fig.update_traces(contours_z=dict(show=True, usecolormap=True, highlightcolor="limegreen", project_z=True)) # xy平面にzの値に対応したcountoursを射影

    # グラフのラベル更新
    fig.update_layout(
        scene=dict(
            xaxis=dict(title=target_x_name),
            yaxis=dict(title=target_y_name),
            zaxis=dict(title=target_z_name),
        )
    )
    # カラーバーの長さを0.5に変更
    fig.update_layout(coloraxis=dict(colorbar=dict(len=0.5)))
    # 余白を削除する
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))

    fig.update_layout(coloraxis={"colorscale": "turbo"})

    # list_var = df[f"{hetero_dist_prefix}_{varname}"].unique()
    # list_var.sort()
    # list_g_c = []
    # list_z = []
    # for var in list_var:
    #     t_g_c = df[df[f"{hetero_dist_prefix}_{varname}"]==var]["g_c"].unique()[0]
    #     list_g_c.append(t_g_c)
    #     list_z.append(0 if is0surface else 1) # surfaceに埋まっちゃうのでちょっと浮かす
    # g_c_line_df = pd.DataFrame.from_dict({xparam_name: list_var, "g": list_g_c, target_z_name: list_z})
    # fig_line = px.line_3d(g_c_line_df, x=xparam_name, y="g", z=target_z_name)
    # fig_line.update_traces(
    #     # marker=dict(
    #     #     color='lightblue',
    #     #     size=10),
    #     line=dict(
    #         color='red',
    #         width=20
    #     )
    # )
    fig_line = plot_gc_line(target_df, target_x_name, z_height=0, gcname="gc")

    # region 引数としてadditional_gc_lineが渡されている場合、追加でプロットする
    if kwargs.get("additional_gc_line", None) is not None:
        """
        ヘテロ性が正規分布に従うとき, 先行研究の1bodyで計算したg_cをプロットする.
        """
        fig_line_additional = plot_gc_line(
            target_df,
            target_x_name,
            z_height=0,
            gcname="additonal gc",
            list_g_c=kwargs.get("additional_gc_line"),
            linecolor="white",
            dash="dash",
        )
        fig_line = concat_figdata(fig_line, fig_line_additional, layout=fig_line.layout)
    # endregion

    if is0surface:
        figconcat_1opacity = concat_figdata(
            concat_figdata(fig, fig0_opacity1), fig_line, layout=fig.layout
        )
        save_mofify_fig(figconcat_1opacity, dump_path / f"{title}_1opacity.html")
        figconcat = concat_figdata(
            concat_figdata(fig, fig0), fig_line, layout=fig.layout
        )
    else:
        figconcat = concat_figdata(fig, fig_line, layout=fig.layout)
    save_mofify_fig(figconcat, dump_path / f"{title}.html")


def plot2d_heatmap(
    target_df,
    target_x_name,
    target_y_name,
    target_z_name,
    title,
    dump_path: Path,
    clipz=None,
    cmaxp=None,
    averageing_method: Literal["mean", "median", None] = None,
    is_zero_mid=False,
    **kwargs,
):
    # font_size = kwargs.get("font_size", 10)
    xarray = np.sort(pd.unique(target_df[target_x_name]))
    yarray = np.sort(pd.unique(target_df[target_y_name]))
    zmatrix = np.empty([yarray.size, xarray.size])
    for i_x, x in enumerate(xarray):
        for i_y, y in enumerate(yarray):
            temp_df = target_df[
                (target_df[target_x_name] == x) & (target_df[target_y_name] == y)
            ]
            assert target_z_name in temp_df.columns, (
                f"target_z_name {target_z_name} is not in target_df columns {temp_df.columns}"
            )
            if temp_df.index.size == 0:
                zmatrix[i_y][i_x] = None
            elif temp_df.index.size == 1:
                zmatrix[i_y][i_x] = temp_df[target_z_name].values[0]
            else:
                if averageing_method == "mean":
                    zmatrix[i_y][i_x] = temp_df[target_z_name].values.mean()
                elif averageing_method == "median":
                    zmatrix[i_y][i_x] = temp_df[target_z_name].values.median()
                elif averageing_method is None:
                    raise ValueError("specify averageing_method")
                else:
                    raise ValueError(
                        f"averageing_method {averageing_method} is not supported"
                    )

    # zmin, zmax = np.min(target_df[target_z_name]), np.max(target_df[target_z_name])
    if clipz is not None:
        zmatrix = np.clip(zmatrix, clipz[0], clipz[1])
    if is_zero_mid:
        heatmap = go.Heatmap(x=xarray, y=yarray, z=zmatrix, colorscale="RdBu_r", zmid=0)
    else:
        heatmap = go.Heatmap(x=xarray, y=yarray, z=zmatrix, coloraxis="coloraxis")
    fig = go.Figure([heatmap])
    if cmaxp is not None:
        assert not is_zero_mid
        cmax = np.percentile(zmatrix, cmaxp)
        title += f"{cmax}"
        fig.update_layout(coloraxis=dict(colorscale="Inferno", cmin=0, cmax=cmax))
    fig_line = plot_gc_line(target_df, target_x_name, z_height=None, gcname="gc")

    # region 引数としてadditional_gc_lineが渡されている場合、追加でプロットする
    if kwargs.get("additional_gc_line", None) is not None:
        """
        ヘテロ性が正規分布に従うとき, 先行研究の1bodyで計算したg_cをプロットする.
        """
        fig_line_additional = plot_gc_line(
            target_df,
            target_x_name,
            z_height=None,
            gcname="additonal gc",
            list_g_c=kwargs.get("additional_gc_line"),
            linecolor="white",
            dash="dash",
        )
        fig_line = concat_figdata(fig_line, fig_line_additional, layout=fig_line.layout)
    # endregion

    figconcat = concat_figdata(fig, fig_line, layout=fig.layout)

    # グラフのラベル更新
    figconcat.update_layout(
        xaxis=dict(title=target_x_name), yaxis=dict(title=target_y_name)
    )

    # カラーバーの長さを0.5に変更
    # figconcat.update_layout(coloraxis=dict(colorbar=dict(len=0.5)))
    # 余白を削除する
    # figconcat.update_layout(margin=dict(l=0, r=0, b=0, t=0))

    if not is_zero_mid:
        figconcat.update_layout(coloraxis={"colorscale": "turbo"})

    save_mofify_fig(figconcat, dump_path / f"{title}.html", is_2d=True)

    fig, ax = plt.subplots(figsize=(6+1, 6))
    if cmaxp is not None:
        c = ax.pcolormesh(xarray, yarray, zmatrix, shading="auto", cmap="turbo", vmin=0, vmax=cmax)
    else:
        c = ax.pcolormesh(xarray, yarray, zmatrix, shading="auto", cmap="turbo")
    fig.colorbar(c, ax=ax)


    list_var = target_df[target_x_name].unique()
    list_g_c = []
    for var in list_var:
        t_g_c = target_df[target_df[target_x_name] == var]["gc"].unique()[0]
        list_g_c.append(t_g_c)
    ax.plot(list_var, list_g_c, color="red", linewidth=3, label="g_c")

    if kwargs.get("additional_gc_line", None) is not None:
        ax.plot(
            list_var,
            kwargs.get("additional_gc_line"),
            color="white",
            linewidth=3,
            linestyle="--",
            label="additonal g_c",
        )

    ax.set_title(title)
    ax.set_xlabel(target_x_name)
    ax.set_ylabel(target_y_name)
    save_fig4paper_plt(fig, dump_path, filename=f"{title}_plt")


def plot3d_mesh(
    target_df,
    target_x_name,
    target_y_name,
    target_z_name,
    title,
    dump_path: Path,
    **kwargs,
):
    font_size = kwargs.get("font_size", 10)
    fig = go.Figure(
        data=[
            go.Mesh3d(
                x=target_df[target_x_name],
                y=target_df[target_x_name],
                z=target_df[target_x_name],
                intensity=target_df[target_x_name],
                colorbar_title="z",
                colorscale=[[0, "gold"], [0.5, "mediumturquoise"], [1, "magenta"]],
            )
        ]
    )

    # グラフのラベル更新
    fig.update_layout(
        scene=dict(
            xaxis=dict(title=target_x_name),
            yaxis=dict(title=target_y_name),
            zaxis=dict(title=target_z_name),
        )
    )
    # グラフ全体とホバーのフォントサイズ変更
    fig.update_layout(font_size=font_size, hoverlabel_font_size=font_size)
    fig.update_layout(yaxis=dict(tickfont=dict(size=5)))
    # カラーバーの長さを0.5に変更
    fig.update_layout(coloraxis=dict(colorbar=dict(len=0.5)))
    # 余白を削除する
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))

    save_mofify_fig(fig, dump_path / f"{title}.html")
