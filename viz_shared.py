import io

import cmcrameri as cmc
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np

CMAP_TARGET = matplotlib.colors.LinearSegmentedColormap.from_list(
    "truncated",
    cmc.cm.broc(np.linspace(.25, .75, 25))
)
CMAP_DEFAULT = "bone_r"
CMAP_BINARY = "binary"
CMAP_TO_CH = {
    "swir": "Greys",
    "nir": "Greys",
    "r": "Reds",
    "g": "Greens",
    "b": "Blues",
    "is_land": CMAP_BINARY,
    "is_cloud": CMAP_BINARY,
    "not_cloud_land": CMAP_BINARY,
}

# Skip `not_cloud_land`
CH_ORDER = [
    "swir", "nir", "r", "g", "b",
    "ndwi_1", "ndwi_2", "ndvi", "gndvi",
    "ndti", "evi", "cari",
    "is_cloud", "is_land"
]


def fig_to_buffer(fig: plt.Figure, format: str, dpi: int) -> io.BytesIO:
    """Convert a Matplotlib figure to a PNG image buffer."""
    buf = io.BytesIO()
    fig.savefig(buf, format=format, dpi=dpi)
    buf.seek(0)
    return buf
