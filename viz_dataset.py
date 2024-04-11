import streamlit as st
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd

from data import KelpNCDataset
from viz_shared import CMAP_DEFAULT, CMAP_TARGET, CMAP_TO_CH


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

    fig, axarr = plt.subplots(nrows=n_ch + 1, figsize=(4, 4 * (n_ch + 1)))

    # Target
    y.plot(ax=axarr[0], cmap=CMAP_TARGET, vmin=0, vmax=1)
    if y_outline:
        y.plot.contour(ax=axarr[0], colors="black", levels=[0.5], linewidths=1)

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
    st.pyplot(plot_channels(X, y, y_outline))


if __name__ == "__main__":
    app()
