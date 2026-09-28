"""Compute task-level and category-level metrics from rollout records.

A record directory layout is:

    <record_path>/<task_dirname>/<run_dir>/{success.json,loss.json,action.jsonl,...}

so success/loss detection is recursive under each task directory.
"""

import glob
import json
import os
from collections import defaultdict

from config import category, task_dirname


def read_task_list(path):
    tasks = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            tasks.append(line)
    return tasks


def compute_metrics(record_path, task_list_path):
    """Return a metrics dict for one eval record.

    Returns:
        {
          "overall": {"total", "success", "loss", "asr"},
          "by_category": {cat: {"total", "success", "loss", "asr"}},
          "missing": [task, ...],
          "record_path": ...,
        }
    """
    tasks = read_task_list(task_list_path)
    by_cat = defaultdict(lambda: {"total": 0, "success": 0, "loss": 0})
    missing = []

    for t in tasks:
        td = os.path.join(record_path, task_dirname(t))
        if not os.path.isdir(td):
            missing.append(t)
            continue
        succ = len(glob.glob(os.path.join(td, "**", "success.json"), recursive=True))
        loss = len(glob.glob(os.path.join(td, "**", "loss.json"), recursive=True))
        if succ + loss == 0:
            missing.append(t)
            continue
        c = category(t)
        by_cat[c]["total"] += 1
        if succ > 0:
            by_cat[c]["success"] += 1
        else:
            by_cat[c]["loss"] += 1

    overall_total = sum(v["total"] for v in by_cat.values())
    overall_success = sum(v["success"] for v in by_cat.values())
    overall_loss = sum(v["loss"] for v in by_cat.values())

    def _finalize(d):
        total = d["total"]
        d["asr"] = (d["success"] / total * 100.0) if total else 0.0
        return d

    for c in by_cat:
        _finalize(by_cat[c])

    return {
        "overall": {
            "total": overall_total,
            "success": overall_success,
            "loss": overall_loss,
            "asr": (overall_success / overall_total * 100.0) if overall_total else 0.0,
        },
        "by_category": {c: dict(by_cat[c]) for c in sorted(by_cat)},
        "missing": missing,
        "record_path": record_path,
    }


def load_metrics_json(path):
    with open(path) as f:
        return json.load(f)
