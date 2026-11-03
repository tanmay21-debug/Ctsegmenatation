# CTSegmentation — CPU-first 2D/3D segmentation

A modular Python toolkit for segmenting 2D images and 3D CT volumes.

## Algorithms included

- `edge_morphology` — morphological-gradient edge mask.
- `gmm` — intensity clustering with Gaussian mixture models.
- `kmeans` — intensity clustering with MiniBatchKMeans.
- `multiotsu` — histogram-based multi-class thresholding.
- `region_growing` — connected region grown from a center seed.

Watershed and local porosity have been removed from this project. They are not registered, selectable in the CLI, or part of the runner.

## Windows PowerShell setup

From the extracted project folder:

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

## Input conventions

- 2D: one image path, such as TIFF, PNG, or a 2D `.npy` array.
- 3D: multi-page TIFF, a `.npy` array shaped `(depth, height, width)`, or a folder of TIFF slices.
- Directory slices are sorted lexicographically; use zero-padded names such as `slice_0001.tif`.

## Run

Interactive:

```powershell
python -m ctseg
```

Non-interactive example:

```powershell
python -m ctseg run --dimension 3d --input "D:\data\scan.tif" --algorithm multiotsu --output "runs\scan_multiotsu"
```

Available algorithm names: `edge_morphology`, `gmm`, `kmeans`, `multiotsu`, `region_growing`.

## Outputs

Each run creates a folder containing `labels.npy`, `report.json`, `label_stats.csv`, and `views/` images. For 3D inputs, the visualization folder includes axial, coronal, and sagittal middle-slice views and overlays. Optional TIFF label export is available with `--export-tiff`.

Compare completed runs:

```powershell
python -m ctseg compare --runs "runs\scan_multiotsu" "runs\scan_kmeans" --output "runs\comparison.csv"
```

## Limitations

Region growing uses a center seed by default and may not target the intended structure. Intensity-based methods can over- or under-segment; inspect outputs and tune parameters for the dataset. This is not the best program to get the work done early as this program is written for memory efficiency over time efficiency.
