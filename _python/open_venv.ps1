Write-Host "Starting script..."
Push-Location

$rootdir = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent

# Virtual environment info 
$venvScripts = resolve-path "..\.venv\Scripts"
$activateScript = Join-Path $venvScripts "Activate.ps1"
# Environment variables
$env:PATH = "$venvScripts;$rootdir\src\.venv\Lib\site-packages\PySide6;$env:PATH"


Set-Location $rootdir
# Start Windows Terminal
$wtProcess = Start-Process wt `
    -ArgumentList `
    "--size", "100,50", `
    "pwsh", "-NoExit", "-wd", $rootdir, "-Command", "& '$activateScript'"
 
Pop-Location
