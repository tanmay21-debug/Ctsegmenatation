import argparse
from pathlib import Path
from .registry import ALGORITHMS
from .runner import run_segmentation
from .compare import compare_runs

def _ask(prompt, default=None):
    while True:
        value = input(prompt).strip()
        if value:
            return value
        if default is not None:
            return default

#prints time reports in shell
def _print_timing_breakdown(report):
    timings = report.get("stage_timings_seconds", {})
    print("\nTiming breakdown:")
    labels = [
        ("input_discovery_seconds", "Input discovery"),
        ("input_loading_seconds", "Input loading total (overlaps below)"),
        ("tiff_decoding_seconds", "TIFF/image decoding"),
        ("volume_assembly_seconds", "Volume assembly / disk writes"),
        ("input_validation_seconds", "Input validation"),
        ("histogram_estimation_seconds", "Histogram / threshold estimation"),
        ("label_assignment_seconds", "Label assignment"),
        ("algorithm_total_seconds", "Algorithm total (includes histogram/labels)"),
        ("labels_npy_save_seconds", "Saving labels.npy"),
        ("labels_tiff_export_seconds", "Optional labels.tif export"),
        ("label_statistics_seconds", "Label statistics CSV"),
        ("visualization_generation_seconds", "Visualization generation"),
        ("ground_truth_loading_seconds", "Ground-truth loading"),
        ("ground_truth_evaluation_seconds", "Ground-truth evaluation"),
        ("report_write_seconds", "Writing report.json"),
    ]
    for key, label in labels:
        if key in timings:
            print(f"  {label:<34} {timings[key]:>9.3f} s")
    print(f"  {'Total wall time':<34} {report['wall_time_seconds']:>9.3f} s")

#interactive cli 
def interactive():
    print("\nCTSeg Starter — CPU-first segmentation")
    print("1) 2D image\n2) 3D volume / TIFF stack")
    dimension = "2d" if _ask("Choose data type [1/2]: ") == "1" else "3d"
    input_path = _ask("Input image/file or folder path: ")
    print("\nAlgorithms:")
    names = list(ALGORITHMS)
    for i, name in enumerate(names, 1):
        print(f" {i}. {name} — {ALGORITHMS[name]['description']}")
    while True:
        choice = _ask("Choose algorithm number: ")
        if choice.isdigit() and 1 <= int(choice) <= len(names):
            algorithm = names[int(choice)-1]
            break
        if choice in ALGORITHMS:
            algorithm = choice
            break
        print("Invalid choice; enter a listed number or algorithm name.")
    classes = "auto"
    gt = _ask("Optional ground-truth label file/folder (Enter to skip): ", "")
    out = _ask("Output folder [runs/run_01]: ", "runs/run_01")
    report = run_segmentation(input_path, dimension, algorithm, out, classes=classes, ground_truth=gt or None)
    print(f"\nDone. Total runtime: {report['wall_time_seconds']:.3f}s; algorithm time: {report['algorithm_time_seconds']:.3f}s; peak RAM: {report['peak_process_rss_mb']:.1f} MiB.")
    _print_timing_breakdown(report)
    print(f"Results: {Path(out).resolve()}")
    print("Compare runs later with: python -m ctseg compare --runs <run_folder_1> <run_folder_2> --output comparison.csv")

#main function
def main():
    parser = argparse.ArgumentParser(prog="ctseg", description="Modular CPU-first 2D/3D segmentation toolkit")
    sub = parser.add_subparsers(dest="command")
    #for run command
    run = sub.add_parser("run", help="Run one segmentation")
    run.add_argument("--dimension", choices=["2d", "3d"], required=True)
    run.add_argument("--input", required=True)
    run.add_argument("--algorithm", choices=list(ALGORITHMS), required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--classes", default="auto", help="Class count; defaults to auto-detection (legacy fixed counts are accepted by algorithms that support them).")
    run.add_argument("--ground-truth")
    run.add_argument("--export-tiff", action="store_true", help="Also export labels.tif (extra disk I/O; disabled by default)")
    #for compare command
    cmp = sub.add_parser("compare", help="Compare completed run folders")
    cmp.add_argument("--runs", nargs="+", required=True)
    cmp.add_argument("--output", default="comparison.csv")
    args = parser.parse_args()
    if args.command is None: #to open the interactive cli
        interactive()
    elif args.command == "run":
        report = run_segmentation(args.input, args.dimension, args.algorithm, args.output, args.classes, args.ground_truth, export_tiff=args.export_tiff)
        print(f"Completed {report['algorithm']}: total {report['wall_time_seconds']:.3f}s; algorithm {report['algorithm_time_seconds']:.3f}s; peak RAM {report['peak_process_rss_mb']:.1f} MiB. Report: {Path(args.output) / 'report.json'}")
        _print_timing_breakdown(report)
    elif args.command == "compare":
        rows = compare_runs(args.runs, args.output)
        print(f"Wrote comparison for {len(rows)} run(s) to {Path(args.output).resolve()}")

if __name__ == "__main__":
    main()
