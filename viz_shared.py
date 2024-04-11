import matplotlib.colors
import numpy as np
import cmcrameri as cmc


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
