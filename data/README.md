# Dataset: EuroSAT (RGB)

**What it is:** 27,000 labelled Sentinel-2 satellite image patches (64 × 64 px, RGB), each covering roughly 640 m × 640 m, in 10 land-use / land-cover classes:
AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop, Residential, River, SeaLake (2,000–3,000 images per class).

**Source paper:** Helber, P., Bischke, B., Dengel, A., & Borth, D. (2019). EuroSAT: A novel dataset and deep learning benchmark for land use and land cover classification. *IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 12*(7), 2217–2226. https://doi.org/10.1109/JSTARS.2019.2918242

**Official repository:** https://github.com/phelber/eurosat
**Imagery:** Copernicus Sentinel-2 (ESA), free and open data. The dataset is published for research use (MIT licence on the official repository).
**Not synthetic, not AI-generated.** Real satellite observations from 34 European countries.

## How to get it

The dataset files are **not committed** (≈ 90 MB zip, ≈ 230 MB unpacked). The notebook downloads them automatically on first run:

```python
from torchvision.datasets import EuroSAT
EuroSAT(root="data", download=True)
```

torchvision fetches `EuroSAT.zip` (MD5 `c8fa014336c82ac7804f0398fcb19387`) from the TorchGeo mirror on Hugging Face:
https://huggingface.co/datasets/torchgeo/eurosat/resolve/c877bcd43f099cd0196738f714544e355477f3fd/EuroSAT.zip

**Manual fallback:** download `EuroSAT.zip` (RGB version) from the official repository above, and unzip it so the images sit at `data/eurosat/2750/<ClassName>/<ClassName>_<n>.jpg`. The notebook then loads it without downloading.

## Files in this folder

| File | Purpose | Committed? |
|---|---|---|
| `README.md` | This file | yes |
| `splits.csv` | Stratified 70 / 15 / 15 train / val / test assignment for every image (created by the notebook with seed 42) | yes |
| `eurosat/` | Raw images (auto-downloaded) | no |
| `eurosat_rgb_uint8.npz` | Cached array of all images, created by the notebook to speed up re-runs | no |
