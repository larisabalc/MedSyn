import csv
from transformers import TrainerCallback

class CSVLoggerCallback(TrainerCallback):
    def __init__(self, train_log_file, eval_log_file):
        self.train_log_file = train_log_file
        self.eval_log_file = eval_log_file
        
        with open(self.train_log_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "step", "loss", "learning_rate"])
        with open(self.eval_log_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "eval_loss", "eval_bleu", "other_metrics_if_any"])  

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        if "loss" in logs:
            with open(self.train_log_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([state.epoch, state.global_step, logs.get("loss"), logs.get("learning_rate")])
        if "eval_loss" in logs:
            with open(self.eval_log_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([state.epoch, logs.get("eval_loss"), logs.get("eval_bleu", ""), ""])  