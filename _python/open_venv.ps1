Write-Host "Starting script..."
Push-Location

$rootdir = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$venvScripts = resolve-path "..\.venv\Scripts"
$activateScript = Join-Path $venvScripts "Activate.ps1"

# Add UV to PATH if installed in venv
if (Test-Path "$venvScripts/uv.exe") {
    $env:PATH = "$venvScripts;$env:PATH"
}

Set-Location $rootdir
# Start Windows Terminal with UV environment
$wtProcess = Start-Process wt `
    -ArgumentList `
    "--size", "100,50", `
    "pwsh", "-NoExit", "-wd", $rootdir, "-Command", "& '$activateScript'"
 
Pop-Location
