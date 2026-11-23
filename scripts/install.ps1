# clipagent installer for Windows (PowerShell):
#   powershell -ExecutionPolicy Bypass -File scripts\install.ps1
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

function Say($m) { Write-Host "`n==> $m" -ForegroundColor Cyan }

Say "Checking ffmpeg"
if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
  winget install --id Gyan.FFmpeg -e --accept-source-agreements --accept-package-agreements
  Write-Host "ffmpeg installed. Close and reopen PowerShell so it is on PATH, then run this script again."
  exit 0
}
ffmpeg -version | Select-Object -First 1

Say "Checking Python (3.10+)"
$py = Get-Command py -ErrorAction SilentlyContinue
if (-not $py) {
  winget install --id Python.Python.3.12 -e --accept-source-agreements --accept-package-agreements
  Write-Host "Python installed. Reopen PowerShell and run this script again."
  exit 0
}
py -3 -c "import sys; sys.exit(sys.version_info < (3,10))"
if ($LASTEXITCODE -ne 0) { throw "Python 3.10+ is required (winget install Python.Python.3.12)" }

Say "Creating virtual environment in .venv"
if (-not (Test-Path .venv)) { py -3 -m venv .venv }
& .\.venv\Scripts\python.exe -m pip install --upgrade pip | Out-Null
& .\.venv\Scripts\pip.exe install -e ".[all]"

if (-not (Test-Path .env)) { Copy-Item .env.example .env; Say "Created .env - add your ANTHROPIC_API_KEY there" }

Say "Downloading models"
& .\.venv\Scripts\clipagent.exe setup
Say "Checking everything"
& .\.venv\Scripts\clipagent.exe doctor

Write-Host "`nDone. Next:`n  .\.venv\Scripts\Activate.ps1`n  clipagent serve      # http://localhost:8000"
