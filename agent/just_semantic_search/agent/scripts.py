import os
import platform
import subprocess
import sys
from pathlib import Path

def get_script_path():
    """Get the appropriate script path based on the operating system"""
    package_dir = Path(__file__).parent.parent.parent
    script_name = "parse-papers.bat" if platform.system() == "Windows" else "parse-papers.sh"
    script_path = package_dir / "bin" / script_name
    return str(script_path)

def parse_papers():
    """Wrapper function to execute the parse-papers script"""
    script_path = get_script_path()
    if platform.system() != "Windows":
        # Make sure the script is executable on Unix-like systems
        os.chmod(script_path, 0o755)
    
    try:
        # Pass through all command line arguments to the script
        result = subprocess.run([script_path] + sys.argv[1:], check=True)
        sys.exit(result.returncode)
    except subprocess.CalledProcessError as e:
        print(f"Error executing script: {e}", file=sys.stderr)
        sys.exit(e.returncode)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    parse_papers() 