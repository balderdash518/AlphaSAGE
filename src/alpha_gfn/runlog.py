"""Per-run record: every candidate formula with its training-period metrics and pool decision, the number of
formula evaluations, timing, and the provenance (config, code commit, data snapshot) of the run.
Recording only; nothing here feeds back into mining."""
import csv
import json
import os
import platform
import subprocess
import time
from datetime import datetime

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
METRIC_KEYS = ['ic', 'max_mut_ic']


def git_commit():
    def run(*args):
        return subprocess.run(['git', *args], cwd=REPO, capture_output=True, text=True).stdout.strip()
    return run('rev-parse', 'HEAD'), bool(run('status', '--porcelain', '--untracked-files=no'))


class RunLog:
    def __init__(self):
        self.active = False

    def start(self, run_dir, config, provenance):
        self.active = True
        self.run_dir = run_dir
        os.makedirs(run_dir, exist_ok=True)
        self.n_evals = 0                 # every formula evaluation on the training data, re-evaluations included
        self.n_invalid = 0               # sampled trajectories that are not a valid formula (never evaluated)
        self.candidates = {}             # expr -> row, in order of first evaluation
        self.round = 0                   # current episode
        self.wall0, self.cpu0 = time.time(), time.process_time()
        self.started_at = datetime.now().isoformat(timespec='seconds')
        commit, dirty = git_commit()
        self.meta = {
            'run_name': os.path.basename(run_dir), 'started_at': self.started_at,
            'code_commit': commit, 'code_dirty': dirty, **provenance,
            'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else '',
            'host': platform.node(),
        }
        with open(os.path.join(run_dir, 'config.json'), 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=1, ensure_ascii=False, default=str)

    def record_invalid(self):
        if self.active:
            self.n_invalid += 1

    def record_eval(self, expr_str, metrics, decision, reason):
        if not self.active:
            return
        self.n_evals += 1
        row = self.candidates.get(expr_str)
        if row is None:
            row = {'expr': expr_str, 'first_round': self.round, 'first_eval': self.n_evals, 'n_evals': 0,
                   **{k: metrics.get(k, np.nan) for k in METRIC_KEYS}}
            self.candidates[expr_str] = row
        row['n_evals'] += 1
        if 'decision' in row and reason == 'duplicate_of_pool_member':
            # resampling a formula that is already in the pool says nothing new about it
            return
        row.update(decision=decision, reason=reason)

    def record_decision(self, expr_str, decision, reason):
        if self.active and expr_str in self.candidates:
            self.candidates[expr_str].update(decision=decision, reason=reason)

    def save(self, pool_exprs=(), final=False):
        if not self.active:
            return
        rows = list(self.candidates.values())
        in_pool = set(pool_exprs)
        for r in rows:
            r['in_pool'] = r['expr'] in in_pool
        cols = ['expr', 'first_round', 'first_eval', 'n_evals', *METRIC_KEYS, 'decision', 'reason', 'in_pool']
        with open(os.path.join(self.run_dir, 'candidates.csv'), 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        self.meta.update({
            'finished': final, 'saved_at': datetime.now().isoformat(timespec='seconds'), 'rounds': self.round,
            'n_formula_evals': self.n_evals, 'n_unique_formulas': len(rows), 'n_invalid_samples': self.n_invalid,
            'zoo_size': len(in_pool),
            'wall_seconds': round(time.time() - self.wall0, 1),
            'cpu_seconds': round(time.process_time() - self.cpu0, 1),
            # the run is GPU-bound end to end, so GPU time is reported as the wall time on the GPU
            'gpu_seconds': round(time.time() - self.wall0, 1) if torch.cuda.is_available() else 0,
            'gpu_max_memory_gb': round(torch.cuda.max_memory_allocated() / 2**30, 2) if torch.cuda.is_available() else 0,
        })
        with open(os.path.join(self.run_dir, 'meta.json'), 'w', encoding='utf-8') as f:
            json.dump(self.meta, f, indent=1, ensure_ascii=False)

    def append_registry(self, path, extra):
        row = {**{k: self.meta.get(k) for k in [
            'run_name', 'started_at', 'saved_at', 'code_commit', 'code_dirty', 'snapshot', 'snapshot_sha256',
            'n_formula_evals', 'n_unique_formulas', 'zoo_size', 'wall_seconds', 'cpu_seconds', 'gpu_seconds']},
            **extra, 'run_dir': os.path.abspath(self.run_dir)}
        new = not os.path.exists(path)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, 'a', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)


RUN = RunLog()
