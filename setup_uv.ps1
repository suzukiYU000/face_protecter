$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

$PythonVersion = if ($env:PYTHON_VERSION) { $env:PYTHON_VERSION } else { "3.11" }
$Profile = if ($args.Count -ge 1 -and $args[0]) { $args[0] } elseif ($env:TORCH_PROFILE) { $env:TORCH_PROFILE } else { "auto" }
$WithOnnx = if ($env:WITH_ONNX) { $env:WITH_ONNX } else { "0" }
$UvExtraArgs = @()
$LocalTmpDir = Join-Path $ScriptDir ".tmp"
$LocalPipCacheDir = Join-Path $ScriptDir ".pip-cache"
$ProfileFile = Join-Path $ScriptDir ".uv-profile"

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Error 'uv was not found. Install it first with: powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"'
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
    Write-Error "Invalid profile: $Profile`nAllowed values: cpu / cu126 / cu128 / cu130 / auto"
}

if ($WithOnnx -in @("1", "true", "TRUE")) {
    $UvExtraArgs += @("--extra", "onnx")
    $WithOnnx = "1"
}
else {
    $WithOnnx = "0"
}

New-Item -ItemType Directory -Force -Path $LocalTmpDir | Out-Null
New-Item -ItemType Directory -Force -Path $LocalPipCacheDir | Out-Null
New-Item -ItemType Directory -Force -Path models, Results, runs, (Join-Path runs logs) | Out-Null
$env:TMP = $LocalTmpDir
$env:TEMP = $LocalTmpDir
$env:PIP_CACHE_DIR = $LocalPipCacheDir

Write-Host ""
Write-Host "=== YOLO26 / uv setup ==="
Write-Host "Profile: $Profile"
Write-Host "TMP: $env:TMP"
Write-Host "PIP_CACHE_DIR: $env:PIP_CACHE_DIR"
if ($UvExtraArgs.Count -gt 0) {
    Write-Host "Extra: onnx"
}
Write-Host ""

Write-Host "[1/4] Installing Python $PythonVersion via uv"
uv python install $PythonVersion

Write-Host "[2/4] Syncing dependencies from uv.lock"
uv sync --locked --extra $Profile @UvExtraArgs

@(
    "UV_PROFILE=$Profile"
    "UV_WITH_ONNX=$WithOnnx"
) | Set-Content -Encoding ASCII $ProfileFile

Write-Host "[3/4] Verifying torch runtime"
try {
    uv run --no-sync --extra $Profile @UvExtraArgs -- python verify_torch_env.py
}
catch {
    Write-Host "Torch verification failed, but setup completed. Check the command output above."
}

Write-Host "[4/4] Done"
Write-Host ""
Write-Host "Saved launch profile: $ProfileFile"
Write-Host ""
Write-Host "Recommended next step:"
Write-Host "  .\run_app.ps1"
Write-Host ""
Write-Host "Alternative launch command:"
Write-Host "  uv run --no-sync --extra $Profile $($UvExtraArgs -join ' ') -- streamlit run face_mosaic_streamlit_app.py"
