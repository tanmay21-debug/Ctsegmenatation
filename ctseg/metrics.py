import csv
import threading
import numpy as np
import psutil

def memory_snapshot():
    process = psutil.Process()
    return {"process_rss_bytes": int(process.memory_info().rss), "system_available_bytes": int(psutil.virtual_memory().available)}

#to monitor peak memory during the process
class PeakRSSMonitor:

    def __init__(self, interval_seconds=0.05):
        self.process = psutil.Process()
        self.interval_seconds = interval_seconds
        self.peak_rss_bytes = 0
        self.peak_system_used_bytes = 0
        self._stop = threading.Event()
        self._thread = None

    def _sample(self):
        while not self._stop.is_set():
            try:
                rss = int(self.process.memory_info().rss)
                used = int(psutil.virtual_memory().total - psutil.virtual_memory().available)
                self.peak_rss_bytes = max(self.peak_rss_bytes, rss)
                self.peak_system_used_bytes = max(self.peak_system_used_bytes, used)
            except (psutil.Error, OSError):
                pass
            self._stop.wait(self.interval_seconds)

    def start(self):
        self.peak_rss_bytes = int(self.process.memory_info().rss)
        vm = psutil.virtual_memory()
        self.peak_system_used_bytes = int(vm.total - vm.available)
        self._thread = threading.Thread(target=self._sample, name="ctseg-memory-monitor", daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        try:
            self.peak_rss_bytes = max(self.peak_rss_bytes, int(self.process.memory_info().rss))
        except (psutil.Error, OSError):
            pass
        return {"peak_process_rss_bytes": int(self.peak_rss_bytes),
                "peak_process_rss_mb": round(self.peak_rss_bytes / (1024 ** 2), 3),
                "peak_system_used_bytes": int(self.peak_system_used_bytes)}

#to write label stats in ouput directory
def label_stats(labels, path):
    values, counts = np.unique(labels, return_counts=True)
    total = int(labels.size)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "voxel_count", "fraction"])
        for v, n in zip(values, counts):
            writer.writerow([int(v), int(n), float(n / total if total else 0)])
    return {str(int(v)): int(n) for v, n in zip(values, counts)}

def ground_truth_metrics(pred, truth):
    if pred.shape != truth.shape:
        raise ValueError(f"Ground-truth shape {truth.shape} does not match output shape {pred.shape}.")
    labels = sorted(set(np.unique(pred).tolist()) | set(np.unique(truth).tolist()))
    result = {}
    for lab in labels:
        p = pred == lab
        t = truth == lab
        denom = int(p.sum() + t.sum())
        inter = int(np.logical_and(p, t).sum())
        union = int(np.logical_or(p, t).sum())
        result[str(int(lab))] = {"dice": float(2*inter/denom) if denom else 1.0, "iou": float(inter/union) if union else 1.0}
    result["voxel_accuracy"] = float(np.mean(pred == truth))
    return result
