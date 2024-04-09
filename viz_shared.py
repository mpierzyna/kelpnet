CMAP_TARGET = "PiYG"
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
