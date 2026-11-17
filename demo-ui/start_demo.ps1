$ErrorActionPreference = 'Stop'
$python = Join-Path $PSScriptRoot '..\.venv-maintenance\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw 'Unified environment is missing. Install requirements-dev.txt into .venv-maintenance first.'
}
& $python -m streamlit run (Join-Path $PSScriptRoot 'app.py') --server.address 127.0.0.1 --server.port 8501
