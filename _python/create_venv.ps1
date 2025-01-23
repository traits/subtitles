param()

$ErrorActionPreference = "Stop"
Push-Location $PSScriptRoot

$VENV_DIR = "../.venv"
$VENV_PYTHON = Join-Path $VENV_DIR "Scripts/python.exe"

# Create environment with locked Python version
uv venv --python 3.12 $VENV_DIR

# Install the project and its dependencies
& $VENV_PYTHON -m pip install -e .

Write-Host "Environment built from pyproject.toml"
Pop-Location
