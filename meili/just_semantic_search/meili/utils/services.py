from just_semantic_search.meili.rag import *
from pathlib import Path
import time
import subprocess
import requests

from eliot._output import *
from eliot import start_task


def ensure_meili_is_running(meili_service_dir: Path, host: str = "127.0.0.1", port: int = 7700, old_docker_compose: bool = False) -> bool:
    """Start MeiliSearch container if not running and wait for it to be ready"""
    
    with start_task(action_type="ensure_meili_running") as action:
        # Check if MeiliSearch is already running
        url = f"http://{host}:{port}/health"
        try:
            response = requests.get(url)
            if response.status_code == 200:
                return True
        except requests.exceptions.ConnectionError:
            pass

        action.log(message_type="server is not available, so starting_server", host=host, port=port)

        # Navigate to the services/meili directory
        #meili_service_dir = project_root / "services" / "meili"
        
        with start_action(action_type="docker_cleanup") as cleanup_action:
            # Stop and remove existing container if it exists
            result = subprocess.run(
                ["docker", "compose", "down"] if not old_docker_compose else ["docker-compose", "down"], 
                cwd=meili_service_dir,
                capture_output=True,
                text=True
            )
            cleanup_action.log(
                message_type="docker_compose_down",
                stdout=result.stdout,
                stderr=result.stderr,
                return_code=result.returncode
            )
            
            result = subprocess.run(
                ["docker", "rm", "-f", "meilisearch"], 
                stderr=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                text=True
            )
            cleanup_action.log(
                message_type="docker_remove",
                stdout=result.stdout,
                return_code=result.returncode
            )
        
        # Start the container using docker compose
        with start_action(action_type="docker_startup") as startup_action:
            process = subprocess.Popen(
                ["docker", "compose", "up"],
                cwd=meili_service_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            startup_action.log(message_type="docker_compose_started", pid=process.pid)
            time.sleep(4)

        # Wait for MeiliSearch to be ready
        with start_action(action_type="wait_for_healthy") as health_action:
            max_retries = 30
            for i in range(max_retries):
                try:
                    response = requests.get(url)
                    if response.status_code == 200:
                        health_action.log(
                            message_type="server_healthy",
                            attempts=i+1
                        )
                        return True
                except requests.exceptions.ConnectionError:
                    health_action.log(
                        message_type="health_check_failed",
                        attempt=i+1,
                        remaining_attempts=max_retries-i-1
                    )
                    time.sleep(1)
                    continue
                
            action.log(message_type="server_failed_to_start", host=host, port=port)
            raise RuntimeError("MeiliSearch failed to start") 

"""
def ensure_meili_is_running(project_root: Path, host: str = "127.0.0.1", port: int = 7700) -> bool:
    
    with start_task(action_type="ensure_meili_running") as action:
        # Check if MeiliSearch is already running
        url = f"http://{host}:{port}/health"
        try:
            response = requests.get(url)
            if response.status_code == 200:
                return True
        except requests.exceptions.ConnectionError:
            pass

        action.log(message_type="server is not available, so starting_server", host=host, port=port)

        # Start MeiliSearch in background
        meili_script = project_root / "bin" / "meili.sh"
        
        process = subprocess.Popen(["/bin/bash", str(meili_script)], 
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
        time.sleep(4)

        # Wait for MeiliSearch to be ready
        max_retries = 30
        for i in range(max_retries):
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    return True
            except requests.exceptions.ConnectionError:
                time.sleep(1)
                continue
        action.log(message_type="server is not started", host=host, port=port)
        raise RuntimeError("MeiliSearch failed to start")
"""