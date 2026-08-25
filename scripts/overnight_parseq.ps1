$ErrorActionPreference = "Continue"
$Root = "F:\MyOwnProject\FGO-OCR"
Set-Location $Root

$env:PYTHONUNBUFFERED = "1"
$env:FGO_OCR_FONT = "C:\Windows\Fonts\YuGothM.ttc"
$env:FGO_OCR_RESUME = "1"
$env:FGO_OCR_EPOCHS = "24"
$env:FGO_OCR_LR = "5e-5"
$env:FGO_OCR_BATCH = "16"

$py = Join-Path $Root ".venv\Scripts\python.exe"
$logDir = Join-Path $Root "data"
$bak = Join-Path $Root "models\backup"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
New-Item -ItemType Directory -Force -Path $bak | Out-Null
$log = Join-Path $logDir "overnight.log"
$ts = Get-Date -Format "yyyyMMdd_HHmmss"

function Write-Log([string]$msg) {
    $line = "{0} {1}" -f (Get-Date -Format "o"), $msg
    Add-Content -Path $log -Value $line -Encoding utf8
    Write-Host $line
}

function Invoke-Step([string]$name, [string[]]$argv) {
    Write-Log "START $name $($argv -join ' ')"
    & $py -u @argv 2>&1 | ForEach-Object {
        $_ | Tee-Object -FilePath $log -Append
    }
    Write-Log "END $name exit=$LASTEXITCODE"
}

if (-not (Test-Path $py)) {
    Write-Log "missing $py"
    exit 1
}

Copy-Item (Join-Path $Root "models\parseq.onnx") (Join-Path $bak "parseq_$ts.onnx") -ErrorAction SilentlyContinue
Copy-Item (Join-Path $Root "models\parseq.pt") (Join-Path $bak "parseq_$ts.pt") -ErrorAction SilentlyContinue
Write-Log "backup models\backup\parseq_$ts.*"

Write-Log "overnight resume epochs=$($env:FGO_OCR_EPOCHS) lr=$($env:FGO_OCR_LR) batch=$($env:FGO_OCR_BATCH)"
Invoke-Step "train_parseq" @("-m", "fgo_ocr", "train_parseq")
Invoke-Step "eval_new_names" @("scripts\eval_new_names.py")
Invoke-Step "eval_parseq" @("scripts\eval_parseq.py")
Invoke-Step "eval_tab" @("scripts\eval_real.py", "data\real\tab")
Write-Log "overnight done"