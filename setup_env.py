import os
import subprocess
import sys
import venv
from pathlib import Path

REQUIRED_PYTHON_MAJOR = 3
REQUIRED_PYTHON_MINOR = 10

def check_python_version():
    version = sys.version_info
    if version.major != REQUIRED_PYTHON_MAJOR or version.minor < REQUIRED_PYTHON_MINOR:
        print(f"Python {REQUIRED_PYTHON_MAJOR}.{REQUIRED_PYTHON_MINOR}+ is required. You are using {version.major}.{version.minor}")
        sys.exit(1)

def create_virtualenv(venv_dir="venv"):
    print(f"Creating virtual environment in ./{venv_dir}...")
    venv.create(venv_dir, with_pip=True)
    print(f"Virtual environment created at {venv_dir}/")

def install_dependencies(venv_dir="venv"):
    pip_path = Path(venv_dir) / ("Scripts" if os.name == "nt" else "bin") / "pip"
    python_path = Path(venv_dir) / ("Scripts" if os.name == "nt" else "bin") / "python"

    subprocess.check_call([str(pip_path), "install", "-r", "requirements.txt"])
    subprocess.check_call([str(python_path), "-m", "spacy", "download", "en_core_web_sm"])
    subprocess.check_call([str(python_path), "-m", "spacy", "download", "en_core_web_md"])


def main():
    check_python_version()
    create_virtualenv()
    install_dependencies()
    print("\nSetup complete!")
    print("To activate the environment:")
    if os.name == "nt":
        print("   .\\venv\\Scripts\\activate")
    else:
        print("   source venv/bin/activate")
    print("\nThen you can run the project:")
    print("   python main.py")

if __name__ == "__main__":
    main()
