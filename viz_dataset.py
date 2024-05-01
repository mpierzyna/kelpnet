import streamlit as st
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd

from data import KelpNCDataset
from viz_shared import CMAP_DEFAULT, CMAP_TARGET, CMAP_TO_CH, fig_to_buffer


@st.cache_data
def get_dataset() -> KelpNCDataset:
    df_quality = pd.read_csv("quality.csv", index_col=0)
    df_quality = df_quality.sort_values("kelp_fraction", ascending=False)

    ds = KelpNCDataset(img_nc_path="data_ncf/train_imgs_fe.nc", mask_nc_path="data_ncf/train_masks.ncf")
    ds.imgs = ds.imgs.isel(sample=df_quality.index)
    ds.masks = ds.masks.isel(sample=df_quality.index)

    return ds


def plot_channels(X: xr.DataArray, y: xr.DataArray, y_outline: bool) -> plt.Figure:
    all_channels = X.ch.data
    n_ch = len(all_channels)
    n_ch = 0

    fig, axarr = plt.subplots(nrows=n_ch + 1, figsize=(4, 4 * (n_ch + 1)))
    axarr = [axarr] if n_ch == 0 else axarr

    # Target
    y.plot(ax=axarr[0], cmap=CMAP_TARGET, vmin=0, vmax=1, rasterized=True)
    if y_outline:
        y.plot.contour(ax=axarr[0], colors="black", levels=[0.5], linewidths=1)
    axarr[0].set_aspect("equal")
    axarr[0].set_xlabel("")
    axarr[0].set_ylabel("")
    axarr[0].set_xticks([])
    axarr[0].set_yticks([])

    # Channels
    for ax, c in zip(axarr[1:], all_channels):
        cmap = CMAP_TO_CH.get(c, CMAP_DEFAULT)
        X.sel(ch=c).plot(ax=ax, cmap=cmap, vmin=0, vmax=1, cbar_kwargs={"label": c.upper()})
        ax.set_title("")
        ax.set_xlabel("")
        ax.set_ylabel("")

    return fig


def app():
    ds = get_dataset()

    sample_ids = list(ds.imgs.sample.data)
    sample_id = st.sidebar.selectbox("Select sample", sample_ids)
    i = sample_ids.index(sample_id)
    y_outline = st.sidebar.toggle("Target outline", True)

    X, y = ds[i]
    fig = plot_channels(X, y, y_outline)
    st.pyplot(fig)

    if st.sidebar.toggle("Enable download"):
        st.sidebar.download_button(
            label="Download SVG",
            data=fig_to_buffer(fig, format="svg", dpi=150),
            file_name=f"{sample_id}_viz_dataset.svg",
            mime="image/svg+xml",
        )


if __name__ == "__main__":
    app()
