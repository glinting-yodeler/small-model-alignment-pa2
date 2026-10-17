"""Plot measured journals only; no synthetic or placeholder experiment curves."""
from pathlib import Path
from .io import read_json, read_jsonl


def plot_metrics(paths, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    for path in paths:
        rows = read_jsonl(path)
        if not rows:
            raise ValueError(f"No measured metrics: {path}")
        name = Path(path).parent.name
        marker = Path(path).parent / "best" / "alignment_checkpoint.json"
        best_step = read_json(marker).get("step") if marker.is_file() else None
        for metric in ("loss", "reward_accuracy", "reward_margin"):
            fig, ax = plt.subplots(figsize=(7, 4))
            plotted = False
            for phase in ("train", "validation"):
                values = [r for r in rows if r["phase"] == phase and metric in r]
                if values:
                    ax.plot([r["step"] for r in values], [r[metric] for r in values], label=phase)
                    plotted = True
            if plotted:
                if best_step is not None:
                    ax.axvline(best_step, color="black", linestyle="--", linewidth=1,
                               label=f"best checkpoint ({best_step})")
                ax.set(xlabel="Optimizer update", ylabel=metric.replace("_", " "), title=name + " (unsmoothed)")
                ax.legend()
                fig.tight_layout()
                fig.savefig(out / f"{name}_{metric}.png", dpi=180)
            plt.close(fig)
