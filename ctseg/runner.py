from pathlib import Path
import json
import time
import numpy as np
from .io import load_array, save_labels, cleanup_input_cache
from .metrics import memory_snapshot, PeakRSSMonitor, label_stats, ground_truth_metrics
from .registry import ALGORITHMS
from .visualize import save_visualizations


def run_segmentation(input_path, dimension, algorithm, output_dir, classes="auto", ground_truth=None, export_tiff=False):
    if algorithm not in ALGORITHMS:
        raise ValueError(f"Unknown algorithm {algorithm!r}. Available: {', '.join(ALGORITHMS)}")
    #make output directory
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    run_start = time.perf_counter()
    stages = {}
    memory_monitor = PeakRSSMonitor().start()
    before = memory_snapshot()
    image = load_array(input_path, dimension, cache_dir=out, timings=stages)
    metadata = {"input_path": str(input_path), "dimension": dimension, "shape": list(image.shape), "dtype": str(image.dtype), "algorithm": algorithm, "backend": "CPU"}
    kwargs = {"output_path": str(out / "labels.npy")}
    if algorithm == "multiotsu":
        kwargs["block_slices"] = 1
    t = time.perf_counter()
    #the step which launches the algorithm
    labels, details = ALGORITHMS[algorithm]["module"].segment(image, dimension=dimension, classes=classes, **kwargs)
    algorithm_elapsed = time.perf_counter() - t
    stages["algorithm_total_seconds"] = float(algorithm_elapsed)
    for key in ("histogram_estimation_seconds", "label_assignment_seconds"):
        if key in details:
            stages[key] = float(details[key])
    save_timings = {}
    save_labels(labels, out / "labels", timings=save_timings, export_tiff=export_tiff)
    stages.update(save_timings)
    t = time.perf_counter()
    stats = label_stats(labels, out / "label_stats.csv")
    stages["label_statistics_seconds"] = float(time.perf_counter() - t)
    t = time.perf_counter()
    save_visualizations(image, labels, out / "views")
    stages["visualization_generation_seconds"] = float(time.perf_counter() - t)
    #to check against ground truth
    gt_metrics = None
    if ground_truth:
        gt_timings = {}
        t = time.perf_counter()
        truth_cache_dir = out / "_gt_cache"
        truth = load_array(ground_truth, dimension, cache_dir=truth_cache_dir, timings=gt_timings)
        stages["ground_truth_loading_seconds"] = float(time.perf_counter() - t)
        for key, value in gt_timings.items():
            stages[f"ground_truth_{key}"] = value
        t = time.perf_counter()
        gt_metrics = ground_truth_metrics(labels, truth)
        stages["ground_truth_evaluation_seconds"] = float(time.perf_counter() - t)
        if isinstance(truth, np.memmap) and getattr(truth, "filename", None) and Path(truth.filename).name == "_ctseg_input_cache.npy":
            try:
                truth._mmap.close()
                Path(truth.filename).unlink(missing_ok=True)
                truth_cache_dir.rmdir()
            except OSError:
                pass
    memory_after = memory_snapshot()
    peak_memory = memory_monitor.stop()
    stages["total_before_report_seconds"] = float(time.perf_counter() - run_start)
    report = {
        **metadata,
        "algorithm_label": ALGORITHMS[algorithm]["label"],
        "wall_time_seconds": float(stages["total_before_report_seconds"]),
        "algorithm_time_seconds": float(algorithm_elapsed),
        "stage_timings_seconds": {k: round(float(v), 6) for k, v in stages.items()},
        "memory_before": before,
        "memory_after": memory_after,
        **peak_memory,
        "process_rss_delta_bytes": int(memory_after["process_rss_bytes"] - before["process_rss_bytes"]),
        "output_dtype": str(labels.dtype),
        "output_labels": sorted(int(v) for v in stats),
        "voxel_counts_by_label": stats,
        "algorithm_details": details,
        "ground_truth_path": str(ground_truth) if ground_truth else None,
        "ground_truth_metrics": gt_metrics,
        "tiff_label_export_enabled": bool(export_tiff),
        "notes": [
            "Stage timings include input discovery/decoding/assembly/validation, algorithm, output saving, statistics, visualization and optional ground-truth evaluation.",
            "All algorithms receive a disk-backed labels.npy output path where supported; algorithm-specific intermediates may still use RAM.",
            "Peak process RSS is sampled every 50 ms and can miss very brief spikes.",
            "Dice/IoU/voxel accuracy require ground-truth labels."
        ]
    }
    report_path = out / "report.json"
    t = time.perf_counter()
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    stages["report_write_seconds"] = float(time.perf_counter() - t)
    report["stage_timings_seconds"] = {k: round(float(v), 6) for k, v in stages.items()}
    report["wall_time_seconds"] = float(time.perf_counter() - run_start)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    #cleanup the cache file in disk
    cleanup_input_cache(image, out)
    return report
