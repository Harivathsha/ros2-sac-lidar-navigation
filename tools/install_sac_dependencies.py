"""Install SAC tooling with a hard constraint on the existing PyTorch build."""
from pathlib import Path
import importlib.metadata
import subprocess
import sys
import tempfile


def main():
    import torch
    from packaging.version import Version
    version = importlib.metadata.version("torch")
    numpy_version = importlib.metadata.version("numpy")
    if not Version("2.3") <= Version(version) < Version("3"):
        raise SystemExit(f"Existing torch {version} is outside SAC's supported range [2.3, 3). Nothing was changed.")
    if not Version("1.26") <= Version(numpy_version) < Version("3"):
        raise SystemExit(f"Existing numpy {numpy_version} is outside this project's range [1.26, 3). Nothing was changed.")
    project = Path(__file__).resolve().parents[1]
    # Constraining the exact local-version tag also preserves +cuXXX wheels.
    with tempfile.TemporaryDirectory(prefix="sac_dependencies_") as folder:
        constraint = Path(folder) / "constraints.txt"
        constraint.write_text(f"torch=={version}\nnumpy=={numpy_version}\n")
        subprocess.run([sys.executable, "-m", "pip", "install", "--constraint", str(constraint),
                        "-r", str(project / "src/nav_learning/requirements_rl.txt")], check=True)
    after = subprocess.check_output([sys.executable, "-c", "import importlib.metadata; print(importlib.metadata.version('torch'))"], text=True).strip()
    if after != version:
        raise RuntimeError("Unexpected PyTorch change.")
    print(f"SAC dependencies installed. Preserved torch {version}, numpy {numpy_version}; CUDA available={torch.cuda.is_available()}")


if __name__ == "__main__":
    main()
