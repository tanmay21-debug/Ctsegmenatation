import csv
import json
from pathlib import Path

#to compare the report between different runs
def compare_runs(run_dirs, output_csv):
    rows = []
    for folder in run_dirs:
        report_path = Path(folder) / "report.json"
        if not report_path.exists():
            raise FileNotFoundError(f"Missing report.json in {folder}")
        r = json.loads(report_path.read_text(encoding="utf-8"))
        gt = r.get("ground_truth_metrics") or {}
        dice_vals = [v["dice"] for v in gt.values() if isinstance(v, dict) and "dice" in v]
        iou_vals = [v["iou"] for v in gt.values() if isinstance(v, dict) and "iou" in v]
        has_gt = bool(r.get("ground_truth_path")) and bool(gt)
        before = (r.get("memory_before") or {}).get("process_rss_bytes")
        after = (r.get("memory_after") or {}).get("process_rss_bytes")
        peak = r.get("peak_process_rss_bytes")
        rows.append({
            "run_dir": str(folder),
            "algorithm": r.get("algorithm"),
            "dimension": r.get("dimension"),
            "shape": "x".join(map(str, r.get("shape", []))),
            "wall_time_seconds": r.get("wall_time_seconds", "N/A"),
            "algorithm_time_seconds": r.get("algorithm_time_seconds", r.get("wall_time_seconds", "N/A")),
            "process_rss_before_bytes": before if before is not None else "N/A",
            "process_rss_after_bytes": after if after is not None else "N/A",
            "process_rss_delta_bytes": r.get("process_rss_delta_bytes", "N/A"),
            "peak_process_rss_bytes": peak if peak is not None else "N/A (not recorded by older report)",
            "peak_process_rss_mb": r.get("peak_process_rss_mb", "N/A (not recorded by older report)"),
            "labels": ";".join(map(str, r.get("output_labels", []))),
            "mean_dice": sum(dice_vals) / len(dice_vals) if dice_vals else ("N/A (no ground truth)" if not has_gt else "N/A"),
            "mean_iou": sum(iou_vals) / len(iou_vals) if iou_vals else ("N/A (no ground truth)" if not has_gt else "N/A"),
            "voxel_accuracy": gt.get("voxel_accuracy", "N/A (no ground truth)" if not has_gt else "N/A"),
        })
    out = Path(output_csv)
    if out.suffix.lower() != ".csv":
        out = out.with_suffix(".csv") if out.suffix else out.with_name(out.name + ".csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else ["run_dir"]
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return rows
