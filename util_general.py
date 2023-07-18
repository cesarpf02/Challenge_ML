import os
from pathlib import Path

# Empty and create directory
def create_dir(path_dir):
    if not os.path.exists(path_dir):
        Path(path_dir).mkdir(parents=True, exist_ok=True)