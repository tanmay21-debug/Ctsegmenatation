from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.color import label2rgb

#normalize image intensities
def _display_image(image):
    x = np.asarray(image, dtype=np.float32)
    lo, hi = np.percentile(x, [1, 99])
    if hi <= lo:
        return np.zeros(x.shape, dtype=np.float32)
    return np.clip((x - lo) / (hi - lo), 0, 1)

#saves individual image
def _save_panel(image, labels, path, title, overlay=False):
    img = _display_image(image)
    fig, ax = plt.subplots(figsize=(8, 7))
    if title.lower().startswith("input") or title.lower().endswith(" input (middle slice)"):
        ax.imshow(img, cmap="gray", interpolation="nearest")
    elif overlay:
        ax.imshow(label2rgb(labels, image=img, bg_label=0, alpha=0.35, kind="overlay"))
    else:
        ax.imshow(labels, cmap="nipy_spectral", interpolation="nearest")
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)

#fn to save visualizations
def save_visualizations(image, labels, output_dir):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if image.ndim == 2:
        _save_panel(image, image, out / "input.png", "Input image")
        _save_panel(image, labels, out / "labels.png", "Segmentation labels")
        _save_panel(image, labels, out / "overlay.png", "Segmentation overlay", overlay=True)
        return
    #for axial,coronaland sagittal views of the 3d output
    d, h, w = image.shape
    planes = {
        "axial": (image[d//2, :, :], labels[d//2, :, :]),
        "coronal": (image[:, h//2, :], labels[:, h//2, :]),
        "sagittal": (image[:, :, w//2], labels[:, :, w//2]),
    }
    for name, (img, lab) in planes.items():
        _save_panel(img, img, out / f"{name}_input.png", f"{name.title()} input (middle slice)")
        _save_panel(img, lab, out / f"{name}_labels.png", f"{name.title()} labels (middle slice)")
        _save_panel(img, lab, out / f"{name}_overlay.png", f"{name.title()} overlay (middle slice)", overlay=True)
