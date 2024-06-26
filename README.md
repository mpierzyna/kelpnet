# KelpNet: Probabilistic Multi-Task Learning for Satellite-Based Kelp Forest Monitoring


## Dataset and preprocessing

**Original dataset.** The original dataset contains satellite imagery from
[Landsat 7](https://www.usgs.gov/landsat-missions/landsat-7) as input data
and collocated binary masks of kelp forest as target data. The input images
contain 5 visual channels and 2 masks. Images and masks have a resolution of
350 x 350 pixels:

- Short-wave infrared (1550nm - 1750nm)
- Near infrared (770nm - 900nm)
- Red (630nm - 690nm)
- Green (520nm - 600nm)
- Blue (450nm - 520nm)
- Cloud cover mask
- Digital elevation model used as land/sea mask

**Landsat 7 SLC issue.** Landsat 7 has issues with its Scan Line Corrector (SLC)
since ~2005 (https://www.usgs.gov/faqs/what-are-landsat-7-slc-gap-mask-files).
That results in invalid values (`NaN`) in form of stripes on some of the images.
These invalid values are filled with OpenCV (see [`step_1a_impute.ipynb`](step_1a_impute.ipynb))

**Remote sensing indices.** The 5 original bands are used to compute the following
remote sensing (RS) indices as additional features (see [`step_1b_preprocess.ipynb`](step_1b_preprocess.ipynb)):

- NDWI1
- NDWI2
- NDVI
- GNDVI
- NDTI
- EVI
- CARI

All indices are added as additional channels to the original imagery.

**Normalization.** After computation of the RS indices, each channel is normalized
to $[0, 1]$ based on its histogram (see [`step_1b_preprocess.ipynb`](step_1b_preprocess.ipynb)).

## Multi-task ensemble learning

Two models are trained on versions of the same dataset to achieve different tasks.
The motivation is to create two models of different complexities -- a simple one and a complex one --yielding better performance when combined compared to their individual
performance. Additionally, the model outputs can be assessed separately by the user for
agreement and sanity, which is viewed as operational/production-use benefit.

For robustness and uncertainty assessment, multiple similar models of both model types
are trained to form an ensemble of models. That allows us to predict a map of kelp
probabilities instead of just a binary mask. The ensemble is generated by training $n=25$ member models, which each see only three randomly drawn channels from the whole
dataset.

### Simple model: tiled classification

[`torch_simple_clf.py`](torch_simple_clf.py)

### Complex model: semantic segmentation

[`torch_deeplabv3.py`](torch_deeplabv3.py)

## Inference

[`step_3a_eval_ens.py`](step_3a_eval_ens.py): Individual inference for each member of
the ensemble for both model types.

[`step_3b_eval_ens_2staged.py`](step_3b_eval_ens_2staged.py): Combine individual
predictions into single prediction.

## Visualisation

## References

- Competition website: https://www.drivendata.org/competitions/255/kelp-forest-segmentation/