# setup_env.ps1
# Builds a reproducible Python 3.11 environment for the AG News term project.
#
# Why 3.11: the plan uses torchtext (basic_english tokenizer), whose final release
# (0.18.0) has no wheels for Python 3.13. 3.11 installs torch 2.3.0 + torchtext
# 0.18.0 cleanly.
#
# This machine has conda (miniconda) but no `py` launcher / standalone 3.11, so we
# use conda to provide the 3.11 interpreter, then pip-install the pinned deps.
#
# SHARING: share this script + requirements.txt (+ environment.yml). Teammates with
# conda run:   powershell -ExecutionPolicy Bypass -File setup_env.ps1
# or:          conda env create -f environment.yml
#
# After it finishes:
#   conda activate agnews-dl
#   python data_pipeline.py

$ErrorActionPreference = "Stop"
$root    = $PSScriptRoot
$envName = "agnews-dl"

Write-Host "== AG News project environment setup (conda, Python 3.11) ==" -ForegroundColor Cyan

# 1) Locate conda.
$conda = (Get-Command conda -ErrorAction SilentlyContinue).Source
if (-not $conda) {
    foreach ($p in @("$env:USERPROFILE\miniconda3\Scripts\conda.exe",
                     "$env:USERPROFILE\anaconda3\Scripts\conda.exe",
                     "$env:LOCALAPPDATA\miniconda3\Scripts\conda.exe")) {
        if (Test-Path $p) { $conda = $p; break }
    }
}
if (-not $conda) {
    Write-Host "conda not found. Install Miniconda, then re-run:" -ForegroundColor Yellow
    Write-Host "    https://docs.conda.io/en/latest/miniconda.html" -ForegroundColor Yellow
    exit 1
}
Write-Host "Using conda: $conda" -ForegroundColor Green

# 2) Create the env (skip if it already exists).
$exists = (& $conda env list) -match "^\s*$envName\s"
if ($exists) {
    Write-Host "conda env '$envName' already exists - reusing it." -ForegroundColor Yellow
    Write-Host "(Remove with: conda env remove -n $envName  to rebuild from scratch.)" -ForegroundColor Yellow
} else {
    Write-Host "Creating conda env '$envName' with Python 3.11 ..." -ForegroundColor Cyan
    & $conda create -y -n $envName python=3.11
}

# 3) Confirm version, then pip-install pinned dependencies inside the env.
Write-Host "Verifying Python version ..." -ForegroundColor Cyan
& $conda run -n $envName python -c "import sys; print('env Python', '%d.%d.%d' % sys.version_info[:3])"

Write-Host "Upgrading pip and installing requirements.txt ..." -ForegroundColor Cyan
& $conda run -n $envName python -m pip install --upgrade pip setuptools wheel
& $conda run -n $envName python -m pip install -r (Join-Path $root "requirements.txt")

# 4) Done.
Write-Host ""
Write-Host "== Setup complete ==" -ForegroundColor Green
Write-Host "Activate with:" -ForegroundColor Cyan
Write-Host "    conda activate $envName"
Write-Host "Verify the data pipeline with:" -ForegroundColor Cyan
Write-Host "    python data_pipeline.py"
