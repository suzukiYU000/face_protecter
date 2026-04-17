$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir
$ProfileFile = Join-Path $ScriptDir ".uv-profile"
$Profile = "cpu"
$WithOnnx = "0"
$UvExtraArgs = @()

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Error 'uv was not found. Install it first with: powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"'
}

if (-not (Test-Path $ProfileFile)) {
    Write-Error "Launch profile was not found. Run .\setup_uv.ps1 first."
}

foreach ($line in Get-Content $ProfileFile) {
    if ($line -match '^UV_PROFILE=(.+)$') {
        $Profile = $Matches[1].Trim()
    }
    elseif ($line -match '^UV_WITH_ONNX=(.+)$') {
        $WithOnnx = $Matches[1].Trim()
    }
}

if ($WithOnnx -eq "1") {
    $UvExtraArgs += @("--extra", "onnx")
}

uv run --no-sync --extra $Profile @UvExtraArgs -- streamlit run face_mosaic_streamlit_app.py
