param(
    [ValidateSet("smoke", "full")]
    [string]$Mode = "smoke",
    [string]$Profile = "auto",
    [int]$Epochs = 0,
    [int]$Imgsz = 0,
    [int]$Batch = 0,
    [int]$Workers = 0,
    [string]$Model = "models/yolo26l.pt",
    [string]$Name = "",
    [string]$CopyBestTo = "",
    [switch]$NoSync,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraTrainArgs
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

if ($Profile -eq "auto" -and $env:TORCH_PROFILE) {
    $Profile = $env:TORCH_PROFILE
}

$WithOnnx = if ($env:WITH_ONNX) { $env:WITH_ONNX } else { "0" }
$UvExtraArgs = @()
$LocalTmpDir = Join-Path $ScriptDir ".tmp"
$LocalPipCacheDir = Join-Path $ScriptDir ".pip-cache"

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Error "uv was not found. Run .\setup_uv.ps1 first."
}

function Get-RecommendedProfile {
    if ($IsMacOS) {
        return "cpu"
    }

    $nvidia = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if (-not $nvidia) {
        return "cpu"
    }

    $output = & nvidia-smi 2>$null | Out-String
    if ($output -match "CUDA Version:\s*([0-9]+)\.([0-9]+)") {
        $major = [int]$Matches[1]
        $minor = [int]$Matches[2]
        if ($major -ge 13) { return "cu130" }
        if ($major -eq 12 -and $minor -ge 8) { return "cu128" }
        if ($major -eq 12 -and $minor -ge 6) { return "cu126" }
    }

    return "cpu"
}

if ($Profile -eq "auto") {
    $Profile = Get-RecommendedProfile
}

if ($Profile -notin @("cpu", "cu126", "cu128", "cu130")) {
    Write-Error "Invalid profile: $Profile"
}

if ($WithOnnx -in @("1", "true", "TRUE")) {
    $UvExtraArgs += @("--extra", "onnx")
}

New-Item -ItemType Directory -Force -Path $LocalTmpDir | Out-Null
New-Item -ItemType Directory -Force -Path $LocalPipCacheDir | Out-Null
New-Item -ItemType Directory -Force -Path models, Results, runs, (Join-Path runs logs) | Out-Null
$env:TMP = $LocalTmpDir
$env:TEMP = $LocalTmpDir
$env:PIP_CACHE_DIR = $LocalPipCacheDir

if ($Epochs -le 0) {
    $Epochs = if ($Mode -eq "smoke") { 1 } else { 50 }
}

if ($Imgsz -le 0) {
    $Imgsz = if ($Mode -eq "smoke") { 320 } else { 512 }
}

if ($Batch -le 0) {
    $Batch = 4
}

if ([string]::IsNullOrWhiteSpace($Name)) {
    $Name = if ($Mode -eq "smoke") { "smoke" } else { "full_wider_face" }
}

if ([string]::IsNullOrWhiteSpace($CopyBestTo)) {
    $CopyBestTo = if ($Mode -eq "smoke") { "models/yolo26l_face_smoke.pt" } else { "models/yolo26l_face_full.pt" }
}

Write-Host ""
Write-Host "=== YOLO26 / uv training ==="
Write-Host "Mode: $Mode"
Write-Host "Profile: $Profile"
Write-Host "Model: $Model"
Write-Host "Epochs: $Epochs"
Write-Host "Image size: $Imgsz"
Write-Host "Batch: $Batch"
Write-Host "Workers: $Workers"
Write-Host "TMP: $env:TMP"
Write-Host ""

if (-not $NoSync) {
    uv sync --locked --extra $Profile @UvExtraArgs
}

$TrainArgs = @("run")
if ($NoSync) {
    $TrainArgs += "--no-sync"
}
$TrainArgs += @("--extra", $Profile) + $UvExtraArgs + @(
    "--", "python", "train_face_detector.py",
    "--mode", $Mode,
    "--model", $Model,
    "--epochs", "$Epochs",
    "--imgsz", "$Imgsz",
    "--batch", "$Batch",
    "--workers", "$Workers",
    "--name", $Name,
    "--copy-best-to", $CopyBestTo
)

$TrainArgs += $ExtraTrainArgs

uv @TrainArgs
