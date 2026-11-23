"""Read a frozen AlphaData snapshot: its forward-adjusted Qlib export and the split dates in its splits.yaml."""
import os


def alphadata_root():
    return os.path.expanduser(os.environ.get("ALPHADATA_ROOT", "~/alphadata"))


def use_snapshot(snapshot, anchor="last"):
    """Point StockData at the AlphaData export of `snapshot` (AlphaSAGE's original Baostock data is forward-adjusted
    to the download day, so anchor 'last') and return its provenance."""
    qlib_path = os.path.join(alphadata_root(), "exports", snapshot, f"qlib-{anchor}")
    if not os.path.isdir(qlib_path):
        raise FileNotFoundError(f"{qlib_path} not found; run `python -m alphadata export qlib {snapshot} --anchor {anchor}`")
    os.environ["ALPHASAGE_QLIB_PATH"] = qlib_path
    source = open(os.path.join(qlib_path, "SOURCE.txt"), encoding="utf-8").read()
    sha = source.split("sha256 ")[1].split(")")[0] if "sha256 " in source else ""
    return {"snapshot": snapshot, "qlib_path": qlib_path, "snapshot_sha256": sha}


def get_split_dates(snapshot, split):
    """(train, valid, test) date pairs of `split` from the snapshot's frozen splits.yaml."""
    import yaml
    path = os.path.join(alphadata_root(), "snapshots", snapshot, "splits.yaml")
    with open(path, encoding="utf-8") as f:
        splits = yaml.safe_load(f)
    if split not in splits:
        raise KeyError(f"split {split} not in {path}: {list(splits)}")
    s = splits[split]
    return tuple(s["train"]), tuple(s["valid"]), tuple(s["test"])
