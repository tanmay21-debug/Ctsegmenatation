#region_growing
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi


#creates the output file in output directory
def _output(shape, output_path):
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return np.lib.format.open_memmap(path, mode="w+", dtype=np.uint8, shape=shape)
    return np.empty(shape, dtype=np.uint8)


def segment(image, dimension="2d", tolerance_fraction=0.08, output_path=None, **kwargs):
    x = np.asanyarray(image)
    if x.ndim not in (2, 3):
        raise ValueError(f"Expected 2D or 3D input, got {x.shape}")
    #seed is chosen as center of the image/volume
    seed = tuple(s // 2 for s in x.shape)
    seed_value = float(x[seed])
    flat = x.reshape(-1)
    #limit the sample size
    stride = max(1, flat.size // 1_000_000)
    sample = np.asarray(flat[::stride][:1_000_000], dtype=np.float32)
    #estimate the intensity spread
    spread = float(np.percentile(sample, 99) - np.percentile(sample, 1)) if sample.size else 0.0
    #calculate tolerance
    tolerance = max(spread * float(tolerance_fraction), np.finfo(np.float32).eps)
    #create the output file in output directory
    out = _output(x.shape, output_path)
    #if 2d sample
    if x.ndim == 2:
        candidate = np.abs(np.asarray(x, dtype=np.float32) - seed_value) <= tolerance
        #label connected components from candidate array
        grown, _ = ndi.label(candidate, structure=ndi.generate_binary_structure(2, 1))
        seed_label = int(grown[seed])
        out[...] = (grown == seed_label).astype(np.uint8) if seed_label else 0
        return out, {
            "seed_index": list(seed), "seed_intensity": seed_value, "tolerance": tolerance,
            "meaning": "0=outside grown region, 1=connected to center seed",
            "processing_mode": "2D connected component",
        }
    structure2d = ndi.generate_binary_structure(2, 1)
    seed_z = seed[0]

    #function to identify candidate pixels in one slice
    def candidate_plane(z):
        return np.abs(np.asarray(x[z], dtype=np.float32) - seed_value) <= tolerance

    #find the connected components in seed slice
    seed_candidate = candidate_plane(seed_z)
    seed_components, _ = ndi.label(seed_candidate, structure=structure2d)
    seed_label = int(seed_components[seed[1], seed[2]])
    #if seed itself fails candidate test
    if seed_label == 0:
        out[...] = 0
        return out, {
            "seed_index": list(seed), "seed_intensity": seed_value, "tolerance": tolerance,
            "meaning": "0=outside grown region, 1=connected to center seed",
            "processing_mode": "slice-wise 2D labeling with exact adjacent-slice propagation",
            "note": "The center seed did not satisfy the candidate predicate.",
        }
    #writing the selected seed-slice component into the output
    selected = seed_components == seed_label
    out[seed_z] = selected.astype(np.uint8)
    #forward propagating through the volume
    previous = selected
    for z in range(seed_z + 1, x.shape[0]):
        candidate = candidate_plane(z)
        components, _ = ndi.label(candidate, structure=structure2d)
        touching = components[previous & candidate] #checking if any touching pixels exist
        ids = np.unique(touching)
        ids = ids[ids > 0]
        #selecting all the members of the component which is touching the previous slice
        selected = np.isin(components, ids) if ids.size else np.zeros(candidate.shape, dtype=bool) 
        #writing to disk
        out[z] = selected.astype(np.uint8)
        previous = selected
    # Propagating backward independently from the seed slice
    previous = seed_components == seed_label
    for z in range(seed_z - 1, -1, -1):
        candidate = candidate_plane(z)
        components, _ = ndi.label(candidate, structure=structure2d)
        touching = components[previous & candidate]
        ids = np.unique(touching)
        ids = ids[ids > 0]
        selected = np.isin(components, ids) if ids.size else np.zeros(candidate.shape, dtype=bool)
        out[z] = selected.astype(np.uint8)
        previous = selected
    #flush the output
    if isinstance(out, np.memmap):
        out.flush()
    return out, {
        "seed_index": list(seed), "seed_intensity": seed_value, "tolerance": tolerance,
        "meaning": "0=outside grown region, 1=connected to center seed",
        "processing_mode": "slice-wise 2D labeling with exact adjacent-slice propagation (3D 6-connectivity)",
    }
