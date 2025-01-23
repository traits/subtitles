param (
    [string]$file
)

if ( -not $file ) {
    $file = Join-Path $PSScriptRoot "site.txt"
} 

$ErrorActionPreference = "Stop"
Push-Location $PSScriptRoot

$VENV_DIR = "../.venv"
$VENV_SCRIPT_DIR = "$VENV_DIR/Scripts"
$PY_VERSION = "3.12"
$REQFILES_DIR = $PSScriptRoot

py -$PY_VERSION -m venv $VENV_DIR

Push-Location
Set-Location $VENV_SCRIPT_DIR
.\Activate.ps1
Set-Location $REQFILES_DIR

Write-Host "Starting $($MyInvocation.MyCommand.Name) $file"

py -m pip install -r $file
Set-Location $VENV_SCRIPT_DIR
deactivate

Write-Host "Script finished."

Pop-Location
