$ErrorActionPreference = "Stop"

$RunName = "20260410_nonDimm_goodmatchCaseB_excel_nondim_hotwall_no_rxx_cuda_6h_001"
$Python = "D:\Anaconda\envs\stefan_pinn\python.exe"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$OutDir = Join-Path $Root "outputs\$RunName"
$CkptDir = Join-Path $Root "checkpoints\$RunName"

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
New-Item -ItemType Directory -Force -Path $CkptDir | Out-Null

$Stdout = Join-Path $OutDir "train_stdout.log"
$Stderr = Join-Path $OutDir "train_stderr.log"
$RunLog = Join-Path $OutDir "run_launcher.log"
$Latest = Join-Path $CkptDir "latest.pth"

$TrainArgs = @(
    "-B", "train.py",
    "--device", "cuda",
    "--phase1-iters", "1000",
    "--phase2-iters", "285000",
    "--phase3-iters", "0",
    "--phase2-lr-start", "1e-4",
    "--phase2-lr-end", "1e-5",
    "--n-interior", "2048",
    "--n-interface", "1024",
    "--n-boundary", "512",
    "--n-ic", "1024",
    "--r-min-interior", "0.01",
    "--resample-all-every", "200",
    "--checkpoint-every", "5000",
    "--plot-every", "50000",
    "--output-dir", $OutDir,
    "--checkpoint-dir", $CkptDir
)

if (Test-Path $Latest) {
    $TrainArgs += @("--resume", $Latest)
}

"launcher started $(Get-Date -Format o)" | Set-Content $RunLog
"run: $RunName" | Add-Content $RunLog
"powershell pid: $PID" | Add-Content $RunLog
"resume checkpoint present: $(Test-Path $Latest)" | Add-Content $RunLog
"phase2 target iteration: 285000" | Add-Content $RunLog
"checkpoints every 5000 iters; monitor plots every 50000 iters (~hourly from the previous run rate)" | Add-Content $RunLog

& $Python @TrainArgs > $Stdout 2> $Stderr
$TrainExit = $LASTEXITCODE
"train exit code $TrainExit at $(Get-Date -Format o)" | Add-Content $RunLog

if (Test-Path $Latest) {
    Copy-Item -LiteralPath $Latest -Destination (Join-Path $CkptDir "latest_after_run.pth") -Force
    "copied latest checkpoint to latest_after_run.pth" | Add-Content $RunLog

    $PostDir = Join-Path $OutDir "postprocess_latest"
    New-Item -ItemType Directory -Force -Path $PostDir | Out-Null
    & $Python -B postprocess.py --checkpoint $Latest --device cuda --output-dir $PostDir `
        > (Join-Path $OutDir "postprocess_stdout.log") `
        2> (Join-Path $OutDir "postprocess_stderr.log")
    "postprocess exit code $LASTEXITCODE at $(Get-Date -Format o)" | Add-Content $RunLog
} else {
    "no latest checkpoint found for final postprocess" | Add-Content $RunLog
}

"launcher finished $(Get-Date -Format o)" | Add-Content $RunLog
