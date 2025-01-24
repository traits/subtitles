param()

$ErrorActionPreference = "Stop"
Push-Location $PSScriptRoot

$VENV_DIR = "../.venv"
$VENV_PYTHON = Join-Path $VENV_DIR "Scripts/python.exe"

# Create environment with locked Python version
uv venv --python 3.12 $VENV_DIR

# Install the project and its dependencies using uv with preview mode
#$env:UV_PREVIEW=1
# uv pip install -e . --python $VENV_PYTHON
uv sync --project "../"

Write-Host "Environment built from pyproject.toml"
Pop-Location
