import os
import csv


LOG_DIR = os.getenv("LOG_DIR", "../runs")


class MetricLogger:
    """
    Handles logging metrics to a CSV file.
    Creates the file if it doesn't exist. On a genuine resume (start_epoch > 1,
    cf. Trainer/TrainerDiffusion's own start_epoch), appends so the continued
    training's rows follow on from where the checkpoint left off. On a FRESH
    run (the default, resume=False) it overwrites -- re-launching a script
    against an exp_name/exp_dir whose logs/metrics.csv already exists from an
    earlier, unrelated run used to silently APPEND a brand-new epoch-1-onward
    block onto that old history with no marker separating the two sessions,
    so any downstream reader (plot_training_metrics.py, evaluation scripts)
    would see one seemingly-continuous file mixing two different trainings --
    e.g. picking a "best epoch" that actually belongs to a stale prior run
    rather than the current one. Found via a real Kolmogorov Re90 FNO/UNet
    comparison whose numbers made no sense until this was traced down.
    """
    def __init__(self, filepath, fieldnames, resume=False):
        self.filepath = os.path.join(LOG_DIR,filepath)
        self.fieldnames = fieldnames

        # Check if file exists
        file_exists = os.path.isfile(self.filepath)
        # Append only when genuinely resuming a previous run; otherwise
        # start this file fresh even if one already exists (see class
        # docstring -- this used to always append, silently corrupting the
        # history on every non-resumed re-run).
        mode = 'a' if (resume and file_exists) else 'w'
        self.csvfile = open(self.filepath, mode, newline='')
        self.writer = csv.DictWriter(self.csvfile, fieldnames=self.fieldnames)

        # Write header for a new file, or when (re)starting a file fresh.
        if mode == 'w' or not file_exists:
            self.writer.writeheader()

    def log(self, row_dict):
        """
        Write one row of metrics to CSV.
        row_dict: dictionary with keys matching fieldnames
        """
        self.writer.writerow(row_dict)
        self.csvfile.flush()  # ensure data is written immediately

    def close(self):
        """Close the CSV file."""
        self.csvfile.close()