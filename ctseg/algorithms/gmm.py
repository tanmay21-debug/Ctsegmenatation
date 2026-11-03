#gmm
from pathlib import Path
import numpy as np
from sklearn.mixture import GaussianMixture
from .auto_classes import thresholds_from_modes


#create the output file to have the data stored
def _output(shape, output_path):
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return np.lib.format.open_memmap(path, mode="w+", dtype=np.uint16, shape=shape)
    return np.empty(shape, dtype=np.uint16)


def segment(image, dimension="2d", classes="auto", random_state=42, output_path=None, **kwargs):
    x = np.asanyarray(image)
    flat = x.reshape(-1)
    #creates a sample of size 200_000 or less if array is less
    rng = np.random.default_rng(random_state)
    sample_n = min(flat.size, 200_000)
    sample = np.asarray(flat if flat.size <= sample_n else flat[rng.choice(flat.size, sample_n, replace=False)], dtype=np.float32)
    #removes invalid values like Nan or infinite
    sample = sample[np.isfinite(sample)]
    if sample.size == 0:
        out = _output(x.shape, output_path)
        out[...] = 0
        return out, {"classes": 0, "sample_voxels": 0, "note": "No finite sample values."}
    #estimate no of mixture components by otsu's histogram threshoding
    cuts, mode_details = thresholds_from_modes(x, bins=256)
    unique_n = int(np.unique(sample).size)
    requested_components = max(1, min(len(cuts) + 1, unique_n, sample.size))
    # Standardization of sample intensities by zscore
    center = float(np.mean(sample, dtype=np.float64))
    scale = float(np.std(sample, dtype=np.float64))
    if not np.isfinite(scale) or scale < 1e-8:
        out = _output(x.shape, output_path)
        out[...] = 0
        return out, {"classes": 1, "sample_voxels": int(sample.size), "note": "Near-constant sample; returned one label."}
    fit_sample = ((sample - center) / scale).astype(np.float64).reshape(-1, 1)
    #fit the gaussian model from sklearn with retries
    model = None
    failures = []
    chosen_components = requested_components
    for n in range(requested_components, 0, -1):
        for reg in (1e-4, 1e-3, 1e-2):
            candidate = GaussianMixture(
                n_components=n, covariance_type="diag", random_state=random_state,
                reg_covar=reg, n_init=2, max_iter=150
            )
            try:
                candidate.fit(fit_sample)
                model = candidate
                chosen_components = n
                break #if successful model is trained
            except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
                failures.append(f"k={n}, reg_covar={reg}: {exc}")
        if model is not None:
            break
    if model is None: #if all atempts fail
        raise RuntimeError("GMM fitting failed after reducing components and increasing covariance regularization. " + failures[-1])
    #converting the means back to their original intensities
    centers = model.means_.reshape(-1) * scale + center
    #Sort the cluster centers and map the labels according to intensities
    order = np.argsort(centers)
    remap = np.zeros(len(order), dtype=np.uint16)
    for new_id, old_id in enumerate(order):
        remap[old_id] = new_id
    #create the output file in disk
    out = _output(x.shape, output_path)
    out_flat = out.reshape(-1)
    #predict labels in batches until no voxel left and save to disk
    batch = 250_000
    for start in range(0, flat.size, batch):
        end = min(start + batch, flat.size)
        values = np.asarray(flat[start:end], dtype=np.float64)
        values = (values - center) / scale #standardise using same mean ans scale used for fitting
        pred = model.predict(values.reshape(-1, 1))
        out_flat[start:end] = remap[pred]
    #flush the output
    if isinstance(out, np.memmap):
        out.flush()
    return out, {
        "classes": int(chosen_components),
        "class_count_method": "automatic histogram-mode detection with robust GMM fitting",
        "class_estimation": mode_details,
        "sample_voxels": int(sample.size),
        "means": sorted(float(v) for v in centers),
        "covariance_regularization": float(model.reg_covar),
        "fit_retries": len(failures),
        "prediction_batch_voxels": int(batch),
    }
