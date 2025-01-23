param (
    [string]$venv_dir = "../.venv"
)

$ErrorActionPreference = "Stop"
Push-Location $PSScriptRoot

$VENV_SCRIPT_DIR = "$venv_dir/Scripts"
$PY_VERSION = "3.12"

# Create virtual environment
py -$PY_VERSION -m venv $venv_dir

# Activate and install with UV
Push-Location
Set-Location $VENV_SCRIPT_DIR
.\Activate.ps1

# Install UV if not present
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    pip install uv
}

# Install project dependencies
uv pip install .

Write-Host "Virtual environment setup complete"
deactivate
Pop-Location
