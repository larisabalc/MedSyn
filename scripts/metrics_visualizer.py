import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent.parent
metrics_base = BASE_DIR / "diagnosis_engine" / "trained_models"

run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
results_root = BASE_DIR / "results" / f"evaluation_reports_{run_id}"

models = {
    "context": {
        "dir": metrics_base / "context" / "metrics",
        "train_file": "train_context_log.csv",
        "eval_file": "eval_context_log.csv"
    },
    "no_context": {
        "dir": metrics_base / "no_context" / "metrics",
        "train_file": "train_no_context_log.csv",
        "eval_file": "eval_no_context_log.csv"
    }
}

for model_name, cfg in models.items():

    print(f"Processing: {model_name}")

    metrics_dir = cfg["dir"]

    out_dir = results_root / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(metrics_dir / cfg["train_file"])
    eval_df = pd.read_csv(metrics_dir / cfg["eval_file"])

    plt.figure(figsize=(8, 5))
    plt.plot(train_df["step"], train_df["loss"], marker="o")
    plt.title(f"{model_name} - Training Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "training_loss.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(eval_df["epoch"], eval_df["eval_loss"], marker="o")
    plt.title(f"{model_name} - Evaluation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "evaluation_loss.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(train_df["step"], train_df["learning_rate"], marker="o")
    plt.title(f"{model_name} - Learning Rate")
    plt.xlabel("Step")
    plt.ylabel("LR")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "learning_rate.png")
    plt.close()

    fig, ax1 = plt.subplots(figsize=(9, 5))

    ax1.plot(train_df["step"], train_df["loss"], marker="o")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Loss")

    ax2 = ax1.twinx()
    ax2.plot(train_df["step"], train_df["learning_rate"], marker="x", linestyle="--")
    ax2.set_ylabel("Learning Rate")

    plt.title(f"{model_name} - Loss vs LR")
    fig.tight_layout()

    plt.savefig(out_dir / "combined.png")
    plt.close()

    print(f"Saved → {out_dir}")

print(f"\nAll results saved in: {results_root}")