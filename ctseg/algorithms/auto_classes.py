import time
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

#creates blocks
def _blocks(x, block_slices=1):
    if x.ndim == 2:
        yield (slice(None), slice(None)), x
    else:
        for z in range(0, x.shape[0], block_slices):
            sl = (slice(z, min(z + block_slices, x.shape[0])), slice(None), slice(None))
            yield sl, x[sl]

#finds thresholds in the volume by histogram thresholding
def thresholds_from_modes(image, bins=2048, block_slices=1):
    x = np.asanyarray(image)
    t0 = time.perf_counter()
    #finds boundaries
    lo, hi = np.inf, -np.inf
    for _, block in _blocks(x, block_slices):
        b = np.asarray(block)
        if np.issubdtype(b.dtype, np.floating):
            finite = b[np.isfinite(b)]
            if finite.size == 0:
                continue
            lo = min(lo, float(finite.min()))
            hi = max(hi, float(finite.max()))
        else:
            lo = min(lo, float(b.min())); hi = max(hi, float(b.max()))
    #handles invalid output
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        return [], {"histogram_modes": 1, "histogram_bins": 0, "histogram_estimation_seconds": float(time.perf_counter()-t0), "note": "Constant, empty, or non-finite input."}
    #normalizes the resolution
    bins = int(min(max(256, bins), 4096))
    hist = np.zeros(bins, dtype=np.int64)
    scale = bins / (hi - lo) 
    #building the histogram
    valid_count = 0
    for _, block in _blocks(x, block_slices):
        b = np.asarray(block).reshape(-1)
        if np.issubdtype(b.dtype, np.floating):
            b = b[np.isfinite(b)]
        if b.size == 0:
            continue
        idx = np.floor((b.astype(np.float32, copy=False) - np.float32(lo)) * scale).astype(np.int32)
        np.clip(idx, 0, bins - 1, out=idx) #clips the outside boundary values
        hist += np.bincount(idx, minlength=bins)
        valid_count += int(b.size)
    smooth = gaussian_filter1d(hist.astype(np.float64), sigma=2.0) #gaussian smoothing
    #find peaks from scipy
    peak_height = max(float(smooth.max()) * 0.002, 1.0)
    peak_prominence = max(float(smooth.max()) * 0.01, 1.0)
    peaks, _ = find_peaks(smooth, height=peak_height, prominence=peak_prominence, distance=3)
    #find peaks at endpoints which the upper function excludes
    endpoint_peaks = []
    if smooth[0] >= smooth[1] and smooth[0] >= peak_height:
        endpoint_peaks.append(0)
    if smooth[-1] >= smooth[-2] and smooth[-1] >= peak_height:
        endpoint_peaks.append(len(smooth) - 1)
    #merging endpoint peaks with the peaks
    if endpoint_peaks:
        peaks = np.unique(np.concatenate((peaks, np.asarray(endpoint_peaks, dtype=int))))
    if peaks.size == 0:
        peaks = np.array([int(np.argmax(smooth))])
    peaks = np.sort(peaks)
    #finding the intensity cut points
    edges = np.linspace(lo, hi, bins + 1, dtype=np.float64)
    cuts = []
    for left, right in zip(peaks[:-1], peaks[1:]):
        if right - left < 2:
            continue
        valley = int(left + np.argmin(smooth[left:right + 1]))
        cut = float((edges[valley] + edges[valley + 1]) * 0.5)
        if not cuts or cut > cuts[-1]:
            cuts.append(cut)
    return cuts, {"histogram_modes": int(len(cuts) + 1), "histogram_bins": bins,
                  "mode_peak_indices": [int(v) for v in peaks],
                  "threshold_method": "blockwise global histogram peaks and intervening valleys",
                  "threshold_sample_voxels": int(valid_count),
                  "histogram_estimation_seconds": float(time.perf_counter()-t0),
                  "note": "Label count is inferred from intensity modes; detected modes depend on histogram prominence settings."}

def segment_by_modes(image, output_path=None, block_slices=1):
    x = np.asanyarray(image)
    #finding the cuts for thresholding
    cuts, details = thresholds_from_modes(x, block_slices=block_slices)
    t0 = time.perf_counter()
    if output_path:
        labels = np.lib.format.open_memmap(output_path, mode="w+", dtype=np.uint16, shape=x.shape)
    else:
        labels = np.empty(x.shape, dtype=np.uint16)
    #labelling the block using digitize
    for sl, block in _blocks(x, block_slices):
        labels[sl] = np.digitize(block, cuts).astype(np.uint16, copy=False)
    if isinstance(labels, np.memmap):
        labels.flush()
    details["label_assignment_seconds"] = float(time.perf_counter()-t0)
    details["classes"] = int(len(cuts) + 1)
    details["thresholds"] = [float(v) for v in cuts]
    details["block_slices"] = int(block_slices)
    return labels, details
