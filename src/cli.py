#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any


AA_RE = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]+$", re.IGNORECASE)


def _safe_float(x: Any) -> float | None:
    return float(x) if isinstance(x, (int, float)) else None


def _git_rev(repo_dir: Path) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
            text=True,
        ).strip()
        return out
    except Exception:
        return None


def _ensure_executable(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Expected file at: {path}")
    try:
        path.chmod(path.stat().st_mode | 0o111)
    except Exception:
        pass


def read_fasta(path: Path) -> list[tuple[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"FASTA not found: {path}")

    entries: list[tuple[str, str]] = []
    name: str | None = None
    seq_parts: list[str] = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    seq = "".join(seq_parts).replace(" ", "").upper()
                    entries.append((name, seq))
                name = line[1:].strip() or f"seq{len(entries) + 1}"
                seq_parts = []
            else:
                seq_parts.append(line)

    if name is not None:
        seq = "".join(seq_parts).replace(" ", "").upper()
        entries.append((name, seq))

    if not entries:
        raise ValueError(f"No FASTA entries found in {path}")

    return entries


def sanitize_name(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
    return s[:80] if len(s) > 80 else s


def validate_sequence(seq: str) -> None:
    seq = seq.strip().upper()
    if not seq:
        raise ValueError("Empty sequence encountered.")
    if not AA_RE.match(seq):
        raise ValueError(
            "Sequence contains non-standard characters. "
            "Expected only 20 canonical amino acids (ACDEFGHIKLMNPQRSTVWY)."
        )


def _snapshot_prediction_dirs(upstream_dir: Path) -> set[str]:
    return {p.name for p in upstream_dir.glob("prediction_*") if p.is_dir()}


def _pick_newest_prediction_dir(upstream_dir: Path, before: set[str]) -> Path:
    candidates = [
        p
        for p in upstream_dir.glob("prediction_*")
        if p.is_dir() and p.name not in before
    ]
    if not candidates:
        candidates = [p for p in upstream_dir.glob("prediction_*") if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(
            "No prediction_* directory found in upstream folder after running predict.sh"
        )

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _copy_any(patterns: list[str], src_dir: Path, dest_dir: Path) -> list[str]:
    copied: list[str] = []
    for pattern in patterns:
        for p in src_dir.glob(pattern):
            dest = dest_dir / p.name
            if dest.exists():
                continue
            if p.is_dir():
                shutil.copytree(p, dest)
            else:
                shutil.copy2(p, dest)
            copied.append(str(dest))
    return copied


def summarize_seed_dir(seed_dir: Path) -> dict[str, Any]:
    """
    Summarize all samples in a seed directory. Writes:
      - metrics_summary.json
      - best_model.cif (copy of best sample CIF)
    Returns the summary dict.
    """
    agg_files = sorted(seed_dir.glob("*_confidences_aggregated.json"))
    if not agg_files:
        return {"n_samples": 0, "error": "No *_confidences_aggregated.json found"}

    samples: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None

    for agg in agg_files:
        try:
            d = json.loads(agg.read_text())
        except Exception:
            continue

        m = re.search(r"_sample_(\d+)_confidences_aggregated\.json$", agg.name)
        sample_id = int(m.group(1)) if m else None

        avg_plddt = _safe_float(d.get("avg_plddt"))
        ptm = _safe_float(d.get("ptm"))
        iptm = _safe_float(d.get("iptm"))
        disorder = _safe_float(d.get("disorder"))
        has_clash = d.get("has_clash")

        ranking = _safe_float(d.get("sample_ranking_score"))
        if ranking is None:
            ranking = _safe_float(d.get("ranking_score"))

        cif = None
        if sample_id is not None:
            cifs = list(seed_dir.glob(f"*sample_{sample_id}_model.cif"))
            if cifs:
                cif = cifs[0].name

        row = {
            "sample": sample_id,
            "avg_plddt": avg_plddt,
            "ptm": ptm,
            "iptm": iptm,
            "disorder": disorder,
            "has_clash": has_clash,
            "sample_ranking_score": ranking,
            "aggregated_json": agg.name,
            "model_cif": cif,
        }
        samples.append(row)

        def rank_key(r: dict[str, Any]) -> tuple[bool, float, float]:
            return (
                r["sample_ranking_score"] is not None,
                r["sample_ranking_score"]
                if r["sample_ranking_score"] is not None
                else -1e9,
                r["avg_plddt"] if r["avg_plddt"] is not None else -1e9,
            )

        if best is None or rank_key(row) > rank_key(best):
            best = row

    def metric_range(key: str) -> dict[str, float | None]:
        vals = [r[key] for r in samples if isinstance(r.get(key), (int, float))]
        if not vals:
            return {"min": None, "max": None, "mean": None}
        return {
            "min": min(vals),
            "max": max(vals),
            "mean": sum(vals) / len(vals),
        }

    summary = {
        "n_samples": len(samples),
        "best_by": "sample_ranking_score (fallback avg_plddt)",
        "best": best,
        "range": {
            "avg_plddt": metric_range("avg_plddt"),
            "ptm": metric_range("ptm"),
            "iptm": metric_range("iptm"),
            "disorder": metric_range("disorder"),
            "sample_ranking_score": metric_range("sample_ranking_score"),
        },
    }

    (seed_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2))

    if best and best.get("model_cif"):
        src = seed_dir / best["model_cif"]
        dst = seed_dir / "best_model.cif"
        try:
            if src.exists():
                shutil.copy2(src, dst)
        except Exception:
            pass

    return summary


def run_one_sequence(
    *,
    name: str,
    sequence: str,
    out_dir: Path,
    upstream_dir: Path,
) -> dict[str, Any]:
    predict_sh = upstream_dir / "predict.sh"
    _ensure_executable(predict_sh)
    validate_sequence(sequence)

    seq_out = out_dir / sanitize_name(name)
    seq_out.mkdir(parents=True, exist_ok=True)

    before = _snapshot_prediction_dirs(upstream_dir)

    cmd = ["bash", str(predict_sh.resolve()), sequence]
    start = time.time()
    proc = subprocess.run(
        cmd,
        cwd=str(upstream_dir),
        text=True,
        capture_output=True,
    )
    runtime = time.time() - start

    (seq_out / "stdout.txt").write_text(proc.stdout or "")
    (seq_out / "stderr.txt").write_text(proc.stderr or "")

    artifacts: list[str] = []
    artifacts += _copy_any(
        ["base_query_*.json", "msa_*.log", "template_results_*.json"],
        src_dir=upstream_dir,
        dest_dir=seq_out,
    )

    pred_dir = None
    try:
        pred_dir = _pick_newest_prediction_dir(upstream_dir, before)
        pred_dest = seq_out / "prediction"
        if not pred_dest.exists():
            shutil.copytree(pred_dir, pred_dest)
            artifacts.append(str(pred_dest))
    except Exception as e:
        pred_dir = None
        (seq_out / "collection_error.txt").write_text(str(e))

    metrics: dict[str, Any] = {
        "best_plddt": None,
        "best_sample": None,
        "best_cif_relpath": None,
        "n_samples": 0,
    }

    if pred_dir is not None:
        pred_local = seq_out / "prediction"
        seed_dirs = list(pred_local.glob("query_*/seed_*"))

        if seed_dirs:
            seed_dir = seed_dirs[0]
            seed_summary = summarize_seed_dir(seed_dir)

            best = (seed_summary or {}).get("best") or {}
            metrics["best_plddt"] = best.get("avg_plddt")
            metrics["best_sample"] = best.get("sample")
            metrics["n_samples"] = seed_summary.get("n_samples", 0)

            best_cif = seed_dir / "best_model.cif"
            if best_cif.exists():
                metrics["best_cif_relpath"] = str(best_cif.relative_to(seq_out))

            try:
                shutil.copy2(seed_dir / "metrics_summary.json", seq_out / "metrics_summary.json")
                if best_cif.exists():
                    shutil.copy2(best_cif, seq_out / "best_model.cif")
            except Exception:
                pass

    return {
        "name": name,
        "length": len(sequence),
        "runtime_sec": round(runtime, 2),
        "output_dir": str(seq_out),
        "command": " ".join(cmd),
        "returncode": proc.returncode,
        "artifacts": artifacts,
        "metrics": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mac-native protein folding wrapper using OpenFold3-MLX"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    fold = subparsers.add_parser("fold", help="Fold sequences from a FASTA file")
    fold.add_argument("--fasta", type=Path, required=True)
    fold.add_argument("--out", type=Path, required=True)
    fold.add_argument("--upstream", type=Path, default=Path("openfold-3-mlx"))
    fold.add_argument("--max", type=int, default=0, help="Max number of FASTA entries to run (0=all)")

    args = parser.parse_args()

    if args.command != "fold":
        return

    upstream_dir: Path = args.upstream
    if not upstream_dir.exists():
        raise FileNotFoundError(
            f"Upstream repo not found at {upstream_dir}. Run your setup script first."
        )

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    entries = read_fasta(args.fasta)
    if args.max and args.max > 0:
        entries = entries[: args.max]

    start_all = time.time()
    results: list[dict[str, Any]] = []

    for i, (name, seq) in enumerate(entries, start=1):
        print(f"[{i}/{len(entries)}] Folding: {name} (len={len(seq)})")
        res = run_one_sequence(
            name=name,
            sequence=seq,
            out_dir=out_dir,
            upstream_dir=upstream_dir,
        )
        results.append(res)

    total_runtime = time.time() - start_all

    run_metadata = {
        "fasta": str(args.fasta),
        "output_dir": str(out_dir),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_runtime_sec": round(total_runtime, 2),
        "upstream_repo": str(upstream_dir),
        "upstream_commit": _git_rev(upstream_dir),
        "n_sequences": len(results),
        "sequences": results,
    }

    (out_dir / "run.json").write_text(json.dumps(run_metadata, indent=2))

    print(f"\nCompleted {len(results)} sequences in {total_runtime:.2f}s")
    print(f"Results written to: {out_dir}")

    best_result = None
    best_plddt = None

    for r in results:
        plddt = r.get("metrics", {}).get("best_plddt")
        if plddt is not None and (best_plddt is None or plddt > best_plddt):
            best_plddt = plddt
            best_result = r

    if best_result is not None:
        print(
            f"Top sequence: {best_result['name']} | "
            f"Best sample: {best_result['metrics']['best_sample']} | "
            f"avg pLDDT: {best_result['metrics']['best_plddt']:.2f}"
        )


if __name__ == "__main__":
    main()