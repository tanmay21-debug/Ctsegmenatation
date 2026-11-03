#kmeans
from pathlib import Path
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from .auto_classes import thresholds_from_modes

#creates the output array in the output directory
def _output(shape, output_path):
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return np.lib.format.open_memmap(path, mode="w+", dtype=np.uint16, shape=shape)
    return np.empty(shape, dtype=np.uint16)


def segment(image, dimension="2d", classes="auto", random_state=42, output_path=None, **kwargs):
    x = np.asanyarray(image)
    flat = x.reshape(-1) #flatten the array
    #takes a random sample of size 200_000 or less if array is shorter
    rng = np.random.default_rng(random_state)
    sample_n = min(flat.size, 200_000)
    sample = np.asarray(flat if flat.size <= sample_n else flat[rng.choice(flat.size, sample_n, replace=False)], dtype=np.float32)
    #removes invalid values like Nan or infinite
    sample = sample[np.isfinite(sample)]
    if sample.size == 0:   #incase all values in sample are invalid
        out = _output(x.shape, output_path)
        out[...] = 0
        return out, {"classes": 0, "sample_voxels": 0, "note": "No finite sample values."}
    #estimate no of clusters by histogram thresholding
    cuts, mode_details = thresholds_from_modes(x, bins=256)
    n_clusters = max(1, min(len(cuts) + 1, int(np.unique(sample).size), sample.size))
    # using mini batch kmeans from sklern
    model = MiniBatchKMeans(n_clusters=n_clusters, random_state=random_state, batch_size=8192, n_init=3)
    model.fit(sample.reshape(-1, 1))
    #Sort the cluster centers and map the labels according to intensities
    centers = model.cluster_centers_.ravel()
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
        values = np.asarray(flat[start:end], dtype=np.float32).reshape(-1, 1)
        out_flat[start:end] = remap[model.predict(values)]
    #flush the ouput 
    if isinstance(out, np.memmap):
        out.flush()
    return out, {
        "classes": int(n_clusters),
        "class_count_method": "automatic histogram-mode detection",
        "class_estimation": mode_details,
        "sample_voxels": int(sample.size),
        "centers": sorted(float(v) for v in centers),
        "iterations": int(model.n_iter_),
        "prediction_batch_voxels": int(batch),
    }
