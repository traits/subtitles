Push-Location $PSScriptRoot

uv sync --project "../" --link-mode=copy --upgrade

Write-Host "Environment built from pyproject.toml"
Pop-Location
