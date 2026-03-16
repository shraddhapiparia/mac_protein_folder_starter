#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_run(run_dir: Path) -> dict:
    run_json = run_dir / "run.json"
    if not run_json.exists():
        raise FileNotFoundError(f"Could not find {run_json}")
    return json.loads(run_json.read_text())


def build_sequence_df(run_data: dict) -> pd.DataFrame:
    rows = []
    for seq in run_data.get("sequences", []):
        metrics = seq.get("metrics", {}) or {}
        rows.append(
            {
                "name": seq.get("name"),
                "length": seq.get("length"),
                "runtime_sec": seq.get("runtime_sec"),
                "returncode": seq.get("returncode"),
                "best_plddt": metrics.get("best_plddt"),
                "best_sample": metrics.get("best_sample"),
                "n_samples": metrics.get("n_samples"),
                "output_dir": seq.get("output_dir"),
            }
        )
    return pd.DataFrame(rows)


def build_sample_df(run_dir: Path, seq_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in seq_df.iterrows():
        seq_name = row["name"]
        seq_dir = run_dir / seq_name
        metrics_json = seq_dir / "metrics_summary.json"
        if not metrics_json.exists():
            continue

        data = json.loads(metrics_json.read_text())
        best_sample = (data.get("best") or {}).get("sample")

        pred_dir = seq_dir / "prediction"
        seed_dirs = list(pred_dir.glob("query_*/seed_*"))
        if not seed_dirs:
            continue

        seed_dir = seed_dirs[0]
        agg_files = sorted(seed_dir.glob("*_confidences_aggregated.json"))

        for agg in agg_files:
            try:
                d = json.loads(agg.read_text())
            except Exception:
                continue

            sample = None
            name = agg.name
            if "_sample_" in name:
                try:
                    sample = int(name.split("_sample_")[1].split("_")[0])
                except Exception:
                    sample = None

            rows.append(
                {
                    "sequence": seq_name,
                    "sample": sample,
                    "avg_plddt": d.get("avg_plddt"),
                    "ptm": d.get("ptm"),
                    "sample_ranking_score": d.get("sample_ranking_score", d.get("ranking_score")),
                    "is_best": sample == best_sample,
                }
            )

    return pd.DataFrame(rows)


def plot_runtime_vs_length(df: pd.DataFrame, outdir: Path) -> None:
    if df.empty:
        return

    df = df.sort_values("length")
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.plot(df["length"], df["runtime_sec"], marker="o")

    for _, row in df.iterrows():
        ax.annotate(
            row["name"],
            (row["length"], row["runtime_sec"]),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=8,
        )

    ax.set_xlabel("Sequence length (aa)")
    ax.set_ylabel("Runtime (seconds)")
    ax.set_title("OpenFold-MLX runtime vs sequence length")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "runtime_vs_length.png", dpi=300)
    plt.close(fig)


def plot_confidence_by_sequence(sample_df: pd.DataFrame, outdir: Path) -> None:
    if sample_df.empty:
        return

    sequences = list(sample_df["sequence"].dropna().unique())
    fig, ax = plt.subplots(figsize=(8.5, max(4.5, 0.9 * len(sequences))))

    for i, seq in enumerate(sequences):
        sub = sample_df[sample_df["sequence"] == seq].copy()
        y = [i] * len(sub)
        ax.scatter(sub["avg_plddt"], y, s=50)

        best = sub[sub["is_best"] == True]
        if not best.empty:
            ax.scatter(best["avg_plddt"], [i] * len(best), s=120, marker="*", zorder=3)

    ax.set_yticks(range(len(sequences)))
    ax.set_yticklabels(sequences)
    ax.set_xlabel("avg pLDDT across samples")
    ax.set_ylabel("Sequence")
    ax.set_title("Prediction confidence across sampled structures")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "confidence_by_sequence.png", dpi=300)
    plt.close(fig)


def save_summary_table(seq_df: pd.DataFrame, outdir: Path) -> None:
    keep = ["name", "length", "runtime_sec", "best_sample", "best_plddt", "n_samples", "returncode"]
    seq_df[keep].to_csv(outdir / "benchmark_summary.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    run_dir = args.run_dir
    outdir = args.out if args.out else (run_dir / "plots")
    outdir.mkdir(parents=True, exist_ok=True)

    run_data = load_run(run_dir)
    seq_df = build_sequence_df(run_data)
    sample_df = build_sample_df(run_dir, seq_df)

    plot_runtime_vs_length(seq_df, outdir)
    plot_confidence_by_sequence(sample_df, outdir)
    save_summary_table(seq_df, outdir)

    print(f"Saved plots to: {outdir}")


if __name__ == "__main__":
    main()