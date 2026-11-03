#edge_morphology
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu


#create the output file
def _output(shape, output_path):
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return np.lib.format.open_memmap(path, mode="w+", dtype=np.uint8, shape=shape)
    return np.empty(shape, dtype=np.uint8)

#calculates the gradient
def _gradient(block):
    eroded = ndi.grey_erosion(block, size=(3,) * block.ndim)
    dilated = ndi.grey_dilation(block, size=(3,) * block.ndim)
    return dilated - eroded


def segment(image, dimension="2d", output_path=None, block_slices=8, **kwargs):
    x = np.asanyarray(image)
    if x.ndim not in (2, 3):
        raise ValueError(f"Expected 2D or 3D input, got {x.shape}")
    #creates the output file
    out = _output(x.shape, output_path)
    samples = []
    max_sample = 1_000_000
    #2d image is processed all at once
    if x.ndim == 2:
        grad = _gradient(np.asarray(x, dtype=np.float32))
        flat = grad.reshape(-1)
        stride = max(1, flat.size // max_sample)
        sample = flat[::stride][:max_sample]
        #calculate the threshold by otsu from skimage
        threshold = float(threshold_otsu(sample)) if sample.size and sample.min() != sample.max() else (float(sample[0]) if sample.size else 0.0)
        #set all instances of grad>threshold to be 1 and rest 0
        out[...] = (grad > threshold).astype(np.uint8)
        del grad
    else:
        depth = max(1, int(block_slices))
        remaining = max_sample
        #creating a sample by processing block_slices = 8 blocks at a time
        for start in range(0, x.shape[0], depth):
            end = min(start + depth, x.shape[0])
            lo, hi = max(0, start - 1), min(x.shape[0], end + 1)
            slab = np.asarray(x[lo:hi], dtype=np.float32)
            grad = _gradient(slab)
            inner = grad[start - lo:end - lo]
            if remaining > 0: #adding to sample until the max_sample is reached
                stride = max(1, inner.size // max(1, remaining))
                picked = inner.reshape(-1)[::stride][:remaining]
                if picked.size:
                    samples.append(np.array(picked, copy=True))
                    remaining -= picked.size
            del slab, grad
        #calculate the threshold using the sample
        sample = np.concatenate(samples) if samples else np.array([0], dtype=np.float32)
        threshold = float(threshold_otsu(sample)) if sample.min() != sample.max() else float(sample[0])
        #generate the output label and save them to disk based
        for start in range(0, x.shape[0], depth):
            end = min(start + depth, x.shape[0])
            lo, hi = max(0, start - 1), min(x.shape[0], end + 1)
            slab = np.asarray(x[lo:hi], dtype=np.float32)
            grad = _gradient(slab)
            out[start:end] = (grad[start - lo:end - lo] > threshold).astype(np.uint8)
            del slab, grad
    #flush the output
    if isinstance(out, np.memmap):
        out.flush()
    return out, {
        "edge_threshold": threshold,
        "meaning": "0=non-edge, 1=morphological edge",
        "processing_mode": "2D" if x.ndim == 2 else "3D slabwise with one-slice halo",
        "threshold_method": "Otsu on bounded sample of morphological gradients",
    }
