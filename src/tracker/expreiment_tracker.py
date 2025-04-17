import os
import yaml
import mlflow
import sys
from pathlib import Path

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent.parent
sys.path.append(str(PACKAGE_ROOT))

def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "../../config/config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def setup_mlflow(config):
    uri = os.getenv("MLFLOW_TRACKING_URI", config["mlflow"]["tracking_uri"])
    mlflow.set_tracking_uri(uri)
