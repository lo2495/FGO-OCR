$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot\..

$env:FGO_OCR_DATA = if ($env:FGO_OCR_DATA) { $env:FGO_OCR_DATA } else { "D:\FGO-OCR-data" }
$env:FGO_OCR_OUT  = $env:FGO_OCR_DATA
if (-not $env:FGO_OCR_N) { $env:FGO_OCR_N = "160000" }
if (-not $env:FGO_OCR_EPOCHS) { $env:FGO_OCR_EPOCHS = "120" }
if (-not $env:FGO_OCR_LR) { $env:FGO_OCR_LR = "3e-4" }
if (-not $env:FGO_OCR_RESUME) { $env:FGO_OCR_RESUME = "1" }
if (-not $env:FGO_OCR_WORKERS) { $env:FGO_OCR_WORKERS = "4" }
if (-not $env:FGO_OCR_FONT) { $env:FGO_OCR_FONT = "C:\Windows\Fonts\YuGothM.ttc" }

New-Item -ItemType Directory -Force -Path $env:FGO_OCR_DATA | Out-Null
$log = Join-Path $env:FGO_OCR_DATA "overnight_hires.log"
$py = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }

Write-Host "python=$py"
Write-Host "data=$env:FGO_OCR_DATA"
Write-Host "log=$log"
Write-Host "n=$env:FGO_OCR_N epochs=$env:FGO_OCR_EPOCHS"

Start-Transcript -Path $log -Append | Out-Null
try {
    & $py -u scripts\run_hires.py
    $code = $LASTEXITCODE
} finally {
    Stop-Transcript | Out-Null
}
if ($code -ne 0) {
    Write-Host "FAILED exit=$code  看 $log"
    exit $code
}
Write-Host "DONE  log=$log"
