#Multiotsu
from .auto_classes import segment_by_modes

def segment(image, dimension="2d", output_path=None, block_slices=1, **kwargs):
    labels, details = segment_by_modes(image, output_path=output_path, block_slices=block_slices)
    details["method"] = "automatic histogram-mode thresholding"
    details["requested_class_count"] = None
    return labels, details
