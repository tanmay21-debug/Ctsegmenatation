from pathlib import Path
import time
import numpy as np
import tifffile
from skimage import io as skio

SUPPORTED = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".npy"}
SLICE_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}

def load_array(path, dimension, cache_dir=None, timings=None):
    timings = timings if timings is not None else {}
    t_all = time.perf_counter()
    p = Path(path)
    decode_s = 0.0
    assembly_s = 0.0
    discovery_s = 0.0
    if p.is_dir():  #checks if p is a directory
        t = time.perf_counter()
        #sort files so that the slices can be processed in correct order
        files = sorted(x for x in p.iterdir() if x.is_file() and x.suffix.lower() in SLICE_SUFFIXES) 
        discovery_s = time.perf_counter() - t
        if not files:
            raise ValueError(f"No supported image slices found in directory: {p}")
        if dimension == "2d":
            raise ValueError("A directory of slices is 3D input. Choose 3D or select one image file.")
        t = time.perf_counter()
        first = _read_image(files[0]) #reads first image in all slices
        decode_s += time.perf_counter() - t
        first = _to_grayscale(first)
        if first.ndim != 2:
            raise ValueError(f"Expected grayscale 2D slices in a folder; first slice has shape {first.shape}.")
        #create a temp memory file path to directly read from it
        cache_root = Path(cache_dir) if cache_dir else Path.cwd() / "runs"
        cache_root.mkdir(parents=True, exist_ok=True)
        cache_path = cache_root / "_ctseg_input_cache.npy"
        t = time.perf_counter()
        #open_memmap creates the memory map
        arr = np.lib.format.open_memmap(cache_path, mode="w+", dtype=first.dtype, shape=(len(files), *first.shape))
        arr[0] = first #loads the first image in memap
        assembly_s += time.perf_counter() - t
        #for subsequent images to be loaded in memap
        for i, f in enumerate(files[1:], 1):
            t = time.perf_counter()
            sl = _to_grayscale(_read_image(f))
            decode_s += time.perf_counter() - t
            if sl.shape != first.shape or sl.dtype != first.dtype:
                raise ValueError(f"Slice {f.name} has shape/dtype {sl.shape}/{sl.dtype}; expected {first.shape}/{first.dtype}.")
            t = time.perf_counter()
            arr[i] = sl
            assembly_s += time.perf_counter() - t
        t = time.perf_counter()
        arr.flush()
        assembly_s += time.perf_counter() - t
    elif not p.exists():
        raise FileNotFoundError(f"Input does not exist: {p}")
    elif p.suffix.lower() == ".npy":
        t = time.perf_counter()
        arr = np.load(p, mmap_mode="r")
        decode_s = time.perf_counter() - t
    elif p.suffix.lower() in {".tif", ".tiff"}:
        t = time.perf_counter()
        try:
            # TIFFs stored in a directly mappable layout can be opened without copying the whole volume.
            arr = tifffile.memmap(p, mode="r")
        except (ValueError, OSError):
            arr = tifffile.imread(p)
        decode_s = time.perf_counter() - t
    elif p.suffix.lower() in SUPPORTED:
        t = time.perf_counter()
        arr = skio.imread(p)
        decode_s = time.perf_counter() - t
    else:
        raise ValueError(f"Unsupported input type: {p.suffix}")

    t = time.perf_counter()
    #checks if arr is not a memap
    if not isinstance(arr, np.memmap):
        arr = _to_grayscale(np.asarray(arr))
    if dimension == "2d":
        if arr.ndim == 3 and arr.shape[-1] in (3, 4):
            arr = _to_grayscale(arr)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D input but received shape {arr.shape}.")
    else:
        if arr.ndim == 2:
            raise ValueError("Expected 3D input but received a single 2D image.")
        if arr.ndim != 3:
            raise ValueError(f"Expected 3D volume shaped (D,H,W), got {arr.shape}. Convert RGB or multi-channel data first.")
    #rejects non numeric arrays
    if not np.issubdtype(arr.dtype, np.number):
        raise ValueError(f"Input must be numeric; got {arr.dtype}.")
    # Scan float data in blocks; checks if floating values are finite
    if np.issubdtype(arr.dtype, np.floating):
        for block in _blocks(arr):
            if not np.isfinite(block).all():
                raise ValueError("Input contains NaN or infinite values. Clean the input volume before segmentation.")
    validation_s = time.perf_counter() - t
    timings.update({"input_discovery_seconds": float(discovery_s), "tiff_decoding_seconds": float(decode_s),
                    "volume_assembly_seconds": float(assembly_s), "input_validation_seconds": float(validation_s),
                    "input_loading_seconds": float(time.perf_counter() - t_all)})
    return arr

def _read_image(path):
    return tifffile.imread(path) if path.suffix.lower() in {".tif", ".tiff"} else skio.imread(path)

#This produces a 2D image as one block, or yields slices of a 3D volume in groups of block_slices
def _blocks(arr, block_slices=1):
    if arr.ndim == 2:
        yield arr
    else:
        for z in range(0, arr.shape[0], block_slices):
            yield arr[z:min(z + block_slices, arr.shape[0])]

#to grayscale an rgb or gbr file
def _to_grayscale(arr):
    if arr.ndim == 3 and arr.shape[-1] in (3, 4):
        rgb = arr[..., :3].astype(np.float32)
        return np.round(0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]).astype(arr.dtype)
    return arr

#to save the labels as a .npy or .tif in disk
def save_labels(labels, path, timings=None, export_tiff=False):
    t0 = time.perf_counter()
    path = Path(path)
    npy_path = path.with_suffix(".npy")
    backing = getattr(labels, "filename", None)
    if backing is None or Path(backing).resolve() != npy_path.resolve():
        np.save(npy_path, labels)
    npy_elapsed = time.perf_counter() - t0
    tiff_elapsed = 0.0
    if export_tiff:
        t = time.perf_counter()
        if int(labels.nbytes) > 1_073_741_824:
            (path.parent / "labels_tiff_skipped.txt").write_text(
                f"TIFF export skipped because the label volume is {labels.nbytes / (1024**3):.2f} GiB. Use {npy_path.name} as the full-resolution label volume.\n", encoding="utf-8")
        else:
            try:
                tifffile.imwrite(path.with_suffix(".tif"), labels, photometric="minisblack")
            except Exception as exc:
                (path.parent / "labels_tiff_warning.txt").write_text(str(exc), encoding="utf-8")
        tiff_elapsed = time.perf_counter() - t
    if timings is not None:
        timings["labels_npy_save_seconds"] = float(npy_elapsed)
        timings["labels_tiff_export_seconds"] = float(tiff_elapsed)

#to cleanup the cache file in disk
def cleanup_input_cache(image, output_dir):
    cache_path = Path(output_dir) / "_ctseg_input_cache.npy"
    backing = getattr(image, "filename", None)
    if cache_path.exists() and backing is not None and Path(backing).resolve() == cache_path.resolve():
        try:
            image.flush()
            mmap_obj = getattr(image, "_mmap", None)
            if mmap_obj is not None:
                mmap_obj.close()
            cache_path.unlink(missing_ok=True)
        except OSError:
            pass
