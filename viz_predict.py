from typing import Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
import xarray as xr

import shared
import torch_deeplabv3 as dlv3
from viz_shared import (CH_ORDER, CMAP_DEFAULT, CMAP_TARGET, CMAP_TO_CH,
                        fig_to_buffer)

PANEL_WIDTH = 3


@st.cache_data
def load_data() -> Tuple[np.ndarray, np.ndarray, xr.DataArray, xr.DataArray, xr.DataArray]:
    # True / test
    _, _, ds_test = dlv3.get_dataset(use_channels=None, random_seed=shared.GLOBAL_SEED)
    y_test = ds_test.masks
    X_test = ds_test.imgs.sel(ch=CH_ORDER)

    # Prediction (Segmentation)
    scores, y_hat_aa, _ = joblib.load("ens_dlv3/dev/pred_dlv3_test.joblib")
    y_hat_seg = torch.mean(y_hat_aa.float(), dim=0).numpy()

    # Prediction (Classifier)
    y_hat_clf = joblib.load("./pred_clf_test_agg_aa.joblib")

    # Compute kelp fraction...
    _, h, w = y_test.shape
    y_test_kf = y_test.sum(dim=("i", "j")) / (h * w)
    y_test_kf = y_test_kf.compute()
    kf_order = np.argsort(-y_test_kf).data  # most kf comes first

    # ...and sort all data accordingly
    y_test_kf = y_test_kf[kf_order]
    y_test = y_test[kf_order]
    X_test = X_test[kf_order]
    y_hat_seg = y_hat_seg[kf_order]
    y_hat_clf = y_hat_clf[kf_order]

    return y_hat_seg, y_hat_clf, y_test, X_test, y_test_kf


def plot_pred_vs_true(*, y_pred_seg_i, y_pred_clf_i, y_test,
                      show_true_outline: bool) -> plt.Figure:
    fig_width = PANEL_WIDTH * (11 / 5)
    fig_height = PANEL_WIDTH
    fig, (ax_clf, ax_seg, ax_cb) = plt.subplots(
        ncols=3, figsize=(fig_width, fig_height),
        gridspec_kw={"width_ratios": [5, 5, 1]}
    )
    if show_true_outline:
        for ax in [ax_clf, ax_seg]:
            ax.contour(y_test, colors="black", levels=[.5], linewidths=1)

    # Plot predictions
    ax_clf.imshow(y_pred_clf_i, origin="lower", cmap=CMAP_TARGET, vmin=0, vmax=1)  # clf
    ax_clf.set_title("Classification model")

    im = ax_seg.imshow(y_pred_seg_i, origin="lower", cmap=CMAP_TARGET, vmin=0, vmax=1)  # seg
    ax_seg.set_title("Segmentation model")

    # Colorbar
    fig.colorbar(im, ax=ax_cb, shrink=0.8, label="Kelp probability")
    ax_cb.axis("off")

    # No ticks and labels
    for ax in [ax_seg, ax_clf]:
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticks([])
        ax.set_yticks([])

        # Aspect ratio 1:1
        ax.set_aspect("equal")

    return fig


def plot_channels(X: xr.DataArray) -> plt.Figure:
    all_channels = X.ch.data
    n_ch = len(all_channels)
    n_cols = 5
    n_rows = n_ch // n_cols + 1

    fig_width = PANEL_WIDTH * n_cols
    fig_height = fig_width / n_cols * n_rows * 1.4

    fig, axarr = plt.subplots(
        nrows=n_rows, ncols=n_cols,
        figsize=(fig_width, fig_height),
        gridspec_kw={"hspace": 0.3, "wspace": 0.1}
    )
    for ax, c in zip(axarr.flat, all_channels):
        cmap = CMAP_TO_CH.get(c, CMAP_DEFAULT)
        im = ax.imshow(X.sel(ch=c), cmap=cmap, vmin=0, vmax=1, origin="lower")
        fig.colorbar(
            im, ax=ax, orientation="horizontal", pad=0.05,
            location="top", label=c.upper(),
            shrink=0.9,
        )

        # No ticks and labels
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticks([])
        ax.set_yticks([])

        # Aspect ratio 1:1
        ax.set_aspect("equal")

    # Blank unused axis
    for ax in axarr.flat[n_ch:]:
        ax.axis("off")

    return fig


def app():
    st.title("KelpNet predictions")

    y_pred_seg, y_pred_clf, y_test, X_test, kf = load_data()
    sample_ids = list(kf.sample.data)

    # Select sample
    sample_id = st.sidebar.selectbox("Select sample", sample_ids, format_func=lambda s: f"{s}, kf={kf.sel(sample=s).item() * 100:.2f}%")
    i = sample_ids.index(sample_id)
    y_pred_seg_i = y_pred_seg[i]
    y_pred_clf_i = y_pred_clf[i]
    y_test_i = y_test[i]
    X_test_i = X_test[i]

    # True contour
    show_true_outline = st.sidebar.checkbox("Show true outline", value=True)

    # Use threshold
    use_threshold = st.sidebar.checkbox("Use threshold", value=False)
    if use_threshold:
        y_pred_seg_i = (y_pred_seg_i > 0.5).astype(float)
        y_pred_clf_i = (y_pred_clf_i > 0.5).astype(float)

    # # Shape of images
    # w = y_pred_seg_i.shape[1]

    # # Cropping
    # col_a, col_b = st.columns(2)
    # center_x = col_a.slider("x", 0, 100, value=50, step=10) / 100 * w
    # center_y = col_a.slider("y", 0, 100, value=50, step=10) / 100 * w
    # crop = col_b.slider("Crop size", 0, 100, value=100, step=10) / 100 * w / 2

    # i_min = max(0, int(center_x - crop))
    # i_max = min(w, int(center_x + crop))
    # j_min = max(0, int(center_y - crop))
    # j_max = min(w, int(center_y + crop))

    # y_pred_seg_i = y_pred_seg_i[i_min:i_max, j_min:j_max]
    # y_test_i = y_test_i[i_min:i_max, j_min:j_max]
    # X_test_i = X_test_i[i_min:i_max, j_min:j_max]

    # Plot
    fig_pred_vs_true = plot_pred_vs_true(
        y_pred_seg_i=y_pred_seg_i, y_pred_clf_i=y_pred_clf_i, y_test=y_test_i, show_true_outline=show_true_outline
    )
    fig_channels = plot_channels(X_test_i)
    st.header("Predictions")
    st.pyplot(fig_pred_vs_true)

    st.header("Input")
    st.pyplot(fig_channels)

    # Download buttons for figures (SVG)
    if st.sidebar.toggle("Enable download", False):
        st.sidebar.write("Download figures")
        dpi = st.sidebar.number_input("DPI", min_value=50, max_value=1000, value=150, step=50)
        st.sidebar.download_button(
            label="Download prediction vs. true",
            data=fig_to_buffer(fig_pred_vs_true, format="svg", dpi=dpi),
            file_name=f"kelp_{sample_id}_pred_vs_true.svg",
            mime="image/svg+xml",
        )
        st.sidebar.download_button(
            label="Download channels",
            data=fig_to_buffer(fig_channels, format="svg", dpi=dpi),
            file_name=f"kelp_{sample_id}_channels.svg",
            mime="image/svg+xml",
        )


if __name__ == "__main__":
    app()
