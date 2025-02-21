from datetime import datetime
from pathlib import Path

current_dir = Path(__file__).parent
project_dir = current_dir.parent  # Go up 2 levels from test/meili to project root
data_dir = project_dir / "data"
logs = project_dir / "logs"
tacutopapers_dir = data_dir / "tacutopapers_test_rsids_10k"
meili_service_dir = project_dir / "meili"

# Configure Eliot to output to both stdout and log files
log_file_path = logs / f"manual_meili_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
logs.mkdir(exist_ok=True)  # Ensure logs directory exists

# Create both JSON and rendered log files
json_log = open(f"{log_file_path}.json", "w")
rendered_log = open(f"{log_file_path}.txt", "w")
