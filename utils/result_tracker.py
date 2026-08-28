"""
utils/result_tracker.py
Tracks pipeline results.

Generates:
1. Summary CSV / JSON (per-dataset, per-model-combo accuracy)
2. Text summary
"""
import os
import json
import csv
from collections import defaultdict
from typing import List, Dict, Any, Optional, Tuple


class ResultTracker:
    """
    Accumulates per-sample results and computes aggregate metrics.

    Each result entry:
    {
      "dataset":          str,
      "sample_id":        str,
      "ground_truth":     str | None,
      "prediction":       str,
      "correct":          bool | None,   # None if no ground truth
      "planner_model":    str,
      "ocr_model":        str,
      "captcha_type":     str,
      "tools_used":       list[str],
      "duration_sec":     float,
      "error":            str | None,
    }
    """

    def __init__(self, results_dir: str = "results", filename: str = "summary.json"):
        self.results_dir = results_dir
        self._filename   = filename
        os.makedirs(results_dir, exist_ok=True)
        self._entries:      List[Dict[str, Any]] = []
        self._done_keys:    set = set()   # (dataset, sample_id, planner, ocr)
        self._correct_keys: set = set()   # subset of _done_keys that are correct

    # Resume / rerun helpers
    def load(self, filename: str = None) -> int:
        """Load existing results from a previous run into memory.

        Returns the number of entries loaded.
        """
        path = os.path.join(self.results_dir, filename or self._filename)
        if not os.path.exists(path):
            return 0
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            self._entries = data.get("entries", [])
            # Rebuild lookup sets from loaded entries
            self._done_keys.clear()
            self._correct_keys.clear()
            for e in self._entries:
                key = self._lookup_key(
                    e["dataset"], e["sample_id"], e["planner_model"], e["ocr_model"]
                )
                self._done_keys.add(key)
                if e.get("correct"):
                    self._correct_keys.add(key)
            print(f"[ResultTracker] Loaded {len(self._entries)} existing result(s) from {path}")
            return len(self._entries)
        except Exception as exc:
            print(f"[ResultTracker] Could not load existing results: {exc}")
            return 0

    def _lookup_key(self, dataset: str, sample_id: str,
                    planner: str, ocr: str) -> tuple:
        return (dataset, str(sample_id), planner, ocr)

    def is_done(self, dataset: str, sample_id: str,
                planner: str, ocr: str) -> bool:
        """O(1) check — True if this (dataset, sample, planner, ocr) has a result."""
        return self._lookup_key(dataset, sample_id, planner, ocr) in self._done_keys

    def was_correct(self, dataset: str, sample_id: str,
                    planner: str, ocr: str) -> bool:
        """O(1) check — True if the sample was already solved correctly."""
        return self._lookup_key(dataset, sample_id, planner, ocr) in self._correct_keys

    # Recording
    def record(self, entry: Dict[str, Any]):
        """Add a single result entry and update the O(1) lookup sets."""
        self._entries.append(entry)
        key = self._lookup_key(
            entry["dataset"], entry["sample_id"],
            entry["planner_model"], entry["ocr_model"],
        )
        self._done_keys.add(key)
        if entry.get("correct"):
            self._correct_keys.add(key)

    # Aggregation
    def _combo_key(self, e: Dict) -> str:
        return f"{e['planner_model']}+{e['ocr_model']}"

    def summary(self) -> Dict[str, Any]:
        """
        Returns a nested dict:
          {dataset_name: {combo_key: {"correct": int, "total": int, "accuracy": float}}}
        """
        stats: Dict[str, Dict[str, Dict]] = defaultdict(lambda: defaultdict(
            lambda: {"correct": 0, "total": 0, "accuracy": 0.0}
        ))
        for e in self._entries:
            ds    = e.get("dataset", "unknown")
            combo = self._combo_key(e)
            stats[ds][combo]["total"] += 1
            if e.get("correct") is True:
                stats[ds][combo]["correct"] += 1

        # Compute accuracy
        for ds in stats:
            for combo in stats[ds]:
                c = stats[ds][combo]["correct"]
                t = stats[ds][combo]["total"]
                stats[ds][combo]["accuracy"] = round(c / t, 4) if t else 0.0

        return {ds: dict(combos) for ds, combos in stats.items()}

    #  Persistence
    def save_json(self, filename: str = None, silent: bool = False):
        path = os.path.join(self.results_dir, filename or self._filename)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"entries": self._entries, "summary": self.summary()},
                      fh, indent=2, ensure_ascii=False)
        if not silent:
            print(f"[ResultTracker] Saved JSON → {path}")
        return path

    def save_csv(self, filename: str = None, silent: bool = False):
        if not self._entries:
            return None
        csv_name = (filename or self._filename).replace(".json", ".csv")
        path     = os.path.join(self.results_dir, csv_name)
        # Union of all keys to handle entries with different fields
        all_keys: List[str] = []
        seen: set = set()
        for e in self._entries:
            for k in e.keys():
                if k not in seen:
                    all_keys.append(k)
                    seen.add(k)
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=all_keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self._entries)
        if not silent:
            print(f"[ResultTracker] Saved CSV  → {path}")
        return path

    def print_summary(self):
        summ = self.summary()
        print("\n" + "=" * 70)
        print("PIPELINE EVALUATION SUMMARY")
        print("=" * 70)
        for ds, combos in sorted(summ.items()):
            print(f"\nDataset: {ds}")
            for combo, stats in sorted(combos.items(), key=lambda x: -x[1]["accuracy"]):
                print(f"  {combo:50s}  acc={stats['accuracy']:.2%}  "
                      f"({stats['correct']}/{stats['total']})")
        print("=" * 70)