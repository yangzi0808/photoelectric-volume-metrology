$ErrorActionPreference = "Stop"

$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    throw "Missing .venv. Run: python -m venv .venv; .\.venv\Scripts\python.exe -m pip install -r requirements.txt"
}

Push-Location $PSScriptRoot
try {
    & $python -m unittest discover -s tests -v
    if ($LASTEXITCODE -ne 0) {
        throw "Unit tests failed."
    }

    & $python -m volumecv image-demo --out output\image_demo
    if ($LASTEXITCODE -ne 0) {
        throw "Image demo failed."
    }

    & $python -m volumecv image-benchmark `
        --per-shape 5 --repeats 2 --seed 20261110 `
        --out output\image_benchmark_smoke
    if ($LASTEXITCODE -ne 0) {
        throw "Image benchmark failed."
    }
}
finally {
    Pop-Location
}
