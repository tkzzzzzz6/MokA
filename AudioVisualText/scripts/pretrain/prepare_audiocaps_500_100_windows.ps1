Param(
    [string]$Dataset = "d0rj/audiocaps",
    [int]$TrainN = 500,
    [int]$ValN = 100,
    [string]$OutDir = "AudioCaps",
    [int]$Retry = 2,
    [int]$HfTimeout = 60,
    [string]$HfEndpoint = "",
    [string]$CookiesFromBrowser = "",
    [string]$CookiesFile = "",
    [switch]$SkipDownload
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path "scripts/pretrain/pretrain_audio.sh")) {
    Write-Error "Please run this script from AudioVisualText root directory."
}

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Error "Python not found. Install Python 3.9+ and make sure it is in PATH."
}

$PythonExe = "python"

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )

    & $PythonExe @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $PythonExe $($Arguments -join ' ')"
    }
}

if (-not $env:HF_HUB_ETAG_TIMEOUT) {
    $env:HF_HUB_ETAG_TIMEOUT = "$HfTimeout"
}
if (-not $env:HF_HUB_DOWNLOAD_TIMEOUT) {
    $env:HF_HUB_DOWNLOAD_TIMEOUT = "$HfTimeout"
}

if ($HfEndpoint) {
    Write-Host "[prepare] using explicit HF endpoint: $HfEndpoint"
}

if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    Write-Error "ffmpeg is not found in PATH. Install ffmpeg first, then retry."
}

Invoke-Checked -Arguments @("-m", "pip", "install", "-U", "numpy<2", "huggingface-hub<1.0", "datasets", "yt-dlp")

$PrepareArgs = @(
    "scripts/pretrain/prepare_audiocaps.py",
    "--dataset", $Dataset,
    "--train_n", "$TrainN",
    "--val_n", "$ValN",
    "--out_dir", $OutDir,
    "--retry", "$Retry",
    "--hf_timeout", "$HfTimeout"
)

if ($HfEndpoint) {
    $PrepareArgs += @("--hf_endpoint", $HfEndpoint)
}
if ($CookiesFromBrowser) {
    $PrepareArgs += @("--cookies_from_browser", $CookiesFromBrowser)
}
if ($CookiesFile) {
    $PrepareArgs += @("--cookies", $CookiesFile)
}
if ($SkipDownload) {
    $PrepareArgs += "--skip_download"
}

Invoke-Checked -Arguments $PrepareArgs

Write-Host "[ok] AudioCaps subset ready under $OutDir"
Write-Host "[next] Run: .\scripts\pretrain\pretrain_audio_windows.ps1"
