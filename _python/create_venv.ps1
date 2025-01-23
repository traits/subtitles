param()

$ErrorActionPreference = "Stop"
Push-Location $PSScriptRoot

$VENV_DIR = "../.venv"
$VENV_PYTHON = Join-Path $VENV_DIR "Scripts/python.exe"
$PYPROJECT = Join-Path $PSScriptRoot "pyproject.toml"

# Create environment with locked Python version
uv venv --python 3.12 $VENV_DIR

# Install from pyproject.toml using project metadata
uv pip install `
    --python $VENV_PYTHON `
    --project $PYPROJECT `
    --resolution=lowest-direct `
    --strict

Write-Host "Environment built from pyproject.toml"
Pop-Location
