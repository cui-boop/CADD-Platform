import os
import shutil
from datetime import datetime
import uuid

import pandas as pd


RUNS_DIR = "results/runs"
HISTORY_PATH = "results/run_history.csv"


def create_run_id():
    time_part = datetime.now().strftime("%Y-%m-%d-%H-%M")
    random_part = uuid.uuid4().hex[:6]
    return f"{time_part}_{random_part}"


def create_run_dir(run_id):
    run_dir = os.path.join(RUNS_DIR, run_id)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def copy_file_if_exists(src_path, run_dir):
    if os.path.exists(src_path):
        dst_path = os.path.join(run_dir, os.path.basename(src_path))
        shutil.copy2(src_path, dst_path)
        return dst_path
    return None


def save_run_history(record):
    os.makedirs("results", exist_ok=True)

    record_df = pd.DataFrame([record])

    if os.path.exists(HISTORY_PATH):
        history_df = pd.read_csv(HISTORY_PATH)
        history_df = pd.concat([history_df, record_df], ignore_index=True)
    else:
        history_df = record_df

    history_df.to_csv(HISTORY_PATH, index=False, encoding="utf-8-sig")


def load_run_history():
    if os.path.exists(HISTORY_PATH):
        return pd.read_csv(HISTORY_PATH)
    return pd.DataFrame()


def get_run_dir(run_id):
    return os.path.join(RUNS_DIR, run_id)