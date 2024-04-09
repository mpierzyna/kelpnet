import streamlit as st
import joblib
import torch
import matplotlib.pyplot as plt
import xarray as xr

import torch_deeplabv3 as dlv3
import shared
from viz_shared import CMAP_DEFAULT, CMAP_TO_CH, CH_ORDER


@st.cache_data
def load_data():
    # True / test
    _, _, ds_test = dlv3.get_dataset(use_channels=None, random_seed=shared.GLOBAL_SEED)
    y_test = ds_test.masks
    X_test = ds_test.imgs.sel(ch=CH_ORDER)

    # Prediction (Segmentation)
    scores, y_hat_aa, _ = joblib.load("ens_dlv3/dev/pred_dlv3_test.joblib")
    y_hat_aa_mean = torch.mean(y_hat_aa.float(), dim=0)

    return y_hat_aa_mean, y_test, X_test


def plot_pred_vs_true(*, y_pred_i, y_test_i) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.contour(y_test_i, levels=[0])  # true
    im = ax.imshow(y_pred_i, origin="lower", cmap="PiYG", vmin=0, vmax=1)  # pred
    fig.colorbar(im, ax=ax, shrink=0.8, label="Kelp probability")

    # No ticks and labels
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
    n_cols = 4
    n_rows = n_ch // n_cols + 1

    panel_width = 2
    fig_width = panel_width * n_cols
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
    y_pred, y_test, X_test = load_data()

    # Select sample
    i = st.slider("Select sample", 0, y_test.shape[0] - 1)
    y_pred_i = y_pred[i]
    y_test_i = y_test[i]
    X_test_i = X_test[i]

    # Shape of images
    w = y_pred_i.shape[1]

    # Cropping
    col_a, col_b = st.columns(2)
    center_x = col_a.slider("x", 0, 100, value=50, step=10) / 100 * w
    center_y = col_a.slider("y", 0, 100, value=50, step=10) / 100 * w
    crop = col_b.slider("Crop size", 0, 100, value=100, step=10) / 100 * w / 2

    i_min = max(0, int(center_x - crop))
    i_max = min(w, int(center_x + crop))
    j_min = max(0, int(center_y - crop))
    j_max = min(w, int(center_y + crop))

    y_pred_i = y_pred_i[i_min:i_max, j_min:j_max]
    y_test_i = y_test_i[i_min:i_max, j_min:j_max]
    X_test_i = X_test_i[i_min:i_max, j_min:j_max]

    # Plot
    st.pyplot(
        plot_pred_vs_true(y_pred_i=y_pred_i, y_test_i=y_test_i)
    )
    st.pyplot(
        plot_channels(X_test_i)
    )


if __name__ == "__main__":
    app()
