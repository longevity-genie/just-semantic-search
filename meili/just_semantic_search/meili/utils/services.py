from pathlib import Path
import time
import subprocess
from pycomfort.logging import to_nice_stdout
import requests
import shutil

from eliot._output import *
from eliot import start_action, start_task


def ensure_meili_is_running(meili_service_dir: Path = Path(__file__).parent.parent.parent.parent, host: str = "127.0.0.1", port: int = 7700) -> bool:
    """Start MeiliSearch container if not running and wait for it to be ready using Podman & podman compose,
    falling back to Docker Compose if Podman is not available."""
    
    with start_task(action_type="ensure_meili_running") as action:
        action.log(message_type="podman compose start", message = f"starting podman compose at {meili_service_dir.resolve().absolute()}")
        # Check if MeiliSearch is already running
        url = f"http://{host}:{port}/health"
        try:
            response = requests.get(url)
            if response.status_code == 200:
                return True
        except requests.exceptions.ConnectionError:
            pass

        action.log(message_type="server is not available, so starting_server", host=host, port=port)

        # Determine which compose command to use
        compose_cmd = None
        if shutil.which("podman"):
            action.log(message_type="using_podman_compose")
            compose_cmd = ["podman", "compose"]
            container_runtime = "podman"
        elif shutil.which("docker"):
            # Try new "docker compose" first
            result = subprocess.run(
                ["docker", "compose", "version"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                action.log(message_type="using_docker_compose_new")
                compose_cmd = ["docker", "compose"]
            elif shutil.which("docker-compose"):
                action.log(message_type="using_docker_compose_legacy")
                compose_cmd = ["docker-compose"]
            container_runtime = "docker"
            
            if not compose_cmd:
                action.log(message_type="no_docker_compose_available")
                raise RuntimeError("Docker is installed but no compose command is available")
        else:
            action.log(message_type="no_container_runtime_available")
            raise RuntimeError("Neither podman compose nor docker is installed")

        # Cleanup existing containers
        with start_action(action_type=f"{container_runtime}_cleanup") as cleanup_action:
            result = subprocess.run(
                compose_cmd + ["down"],
                cwd=meili_service_dir,
                capture_output=True,
                text=True
            )
            cleanup_action.log(
                message_type=f"{container_runtime}_compose_down",
                stdout=result.stdout,
                stderr=result.stderr,
                return_code=result.returncode
            )

        # Start the container using the selected compose tool
        with start_action(action_type="container_startup") as startup_action:
            compose_up_cmd = compose_cmd + ["up"]
            startup_action.log(message_type="starting_container", command=compose_up_cmd)
            
            process = subprocess.Popen(
                compose_up_cmd,
                cwd=meili_service_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            startup_action.log(message_type="compose_started", pid=process.pid)
            time.sleep(5)

        # Wait for MeiliSearch to be healthy
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
        
if __name__ == "__main__":
    print("trying compose")
    to_nice_stdout()
    ensure_meili_is_running()