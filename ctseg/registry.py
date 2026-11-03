#it helps to generalize the code for all algo

from .algorithms import edge_morphology, gmm, kmeans, multiotsu, region_growing

ALGORITHMS = {
    "edge_morphology": {"label": "Edge morphology", "module": edge_morphology, "description": "Morphological-gradient edge mask."},
    "gmm": {"label": "Gaussian mixture model (GMM)", "module": gmm, "description": "Intensity clustering with Gaussian mixtures."},
    "kmeans": {"label": "K-means", "module": kmeans, "description": "Intensity clustering with MiniBatchKMeans."},
    "multiotsu": {"label": "Multi-Otsu", "module": multiotsu, "description": "Histogram-based multi-class thresholding."},
    "region_growing": {"label": "Region growing", "module": region_growing, "description": "Connected region grown from the center voxel/pixel."},
}
