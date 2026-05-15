$ErrorActionPreference = "Stop"

$RunName = "20260410_nonDimm_goodmatchCaseB_excel_nondim_hotwall_no_rxx_fresh_2h_001"
$Python = "D:\Anaconda\envs\stefan_pinn\python.exe"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$OutDir = Join-Path $Root "outputs\$RunName"
$CkptDir = Join-Path $Root "checkpoints\$RunName"
$PostDir = Join-Path $Root "outputs\${RunName}_post"
$RunLog = Join-Path $OutDir "run_launcher.log"
$Stdout = Join-Path $OutDir "train_stdout.log"
$Stderr = Join-Path $OutDir "train_stderr.log"

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
New-Item -ItemType Directory -Force -Path $CkptDir | Out-Null
New-Item -ItemType Directory -Force -Path $PostDir | Out-Null

$TrainArgs = @(
    "-B", "train.py",
    "--device", "cuda",
    "--phase1-iters", "1000",
    "--phase2-iters", "200000",
    "--phase3-iters", "0",
    "--phase2-lr-start", "1e-4",
    "--phase2-lr-end", "1e-5",
    "--n-interior", "2048",
    "--n-interface", "1024",
    "--n-boundary", "512",
    "--n-ic", "1024",
    "--r-min-interior", "0.01",
    "--resample-all-every", "200",
    "--checkpoint-every", "1000",
    "--plot-every", "50000",
    "--output-dir", $OutDir,
    "--checkpoint-dir", $CkptDir
)

function Join-Args {
    param([string[]]$Items)
    ($Items | ForEach-Object {
        if ($_ -match '[\s"]') {
            '"' + ($_ -replace '"', '\"') + '"'
        } else {
            $_
        }
    }) -join ' '
}

"fresh 2h run started $(Get-Date -Format o)" | Set-Content $RunLog
"run: $RunName" | Add-Content $RunLog
"no resume checkpoint used" | Add-Content $RunLog
"timeout seconds: 7200" | Add-Content $RunLog

$BatPath = Join-Path $OutDir "run_train.bat"
$CommandLine = '"' + $Python + '" ' + (Join-Args $TrainArgs) + ' > "' + $Stdout + '" 2> "' + $Stderr + '"'
@(
    "@echo off",
    "cd /d ""$Root""",
    $CommandLine
) | Set-Content -Path $BatPath -Encoding ASCII
$StartInfo = New-Object System.Diagnostics.ProcessStartInfo
$StartInfo.FileName = "cmd.exe"
$StartInfo.Arguments = "/c call """ + $BatPath + """"
$StartInfo.WorkingDirectory = $Root
$StartInfo.UseShellExecute = $false
$StartInfo.RedirectStandardOutput = $false
$StartInfo.RedirectStandardError = $false
$StartInfo.CreateNoWindow = $true

$Proc = New-Object System.Diagnostics.Process
$Proc.StartInfo = $StartInfo

[void]$Proc.Start()
$Proc.Id | Set-Content (Join-Path $OutDir "train.pid")
"python pid $($Proc.Id)" | Add-Content $RunLog

$Deadline = (Get-Date).AddSeconds(7200)
while (-not $Proc.HasExited -and (Get-Date) -lt $Deadline) {
    Start-Sleep -Seconds 30
    $Proc.Refresh()
}

$Proc.Refresh()
if (-not $Proc.HasExited) {
    "timeout reached; stopping pid $($Proc.Id) at $(Get-Date -Format o)" | Add-Content $RunLog
        $Proc.Kill()
        Get-Process -Name "python" -ErrorAction SilentlyContinue |
            Where-Object { $_.Path -eq $Python -and $_.StartTime -ge (Get-Date).AddHours(-3) } |
            Stop-Process -Force -ErrorAction SilentlyContinue
    $Proc.WaitForExit()
} else {
    "train exited with code $($Proc.ExitCode) at $(Get-Date -Format o)" | Add-Content $RunLog
}

$Latest = Join-Path $CkptDir "latest.pth"
if (Test-Path $Latest) {
    Copy-Item -LiteralPath $Latest -Destination (Join-Path $CkptDir "latest_after_2h.pth") -Force
    "copied latest checkpoint to latest_after_2h.pth" | Add-Content $RunLog

    & $Python -B postprocess.py --checkpoint $Latest --device cuda --output-dir $PostDir `
        > (Join-Path $OutDir "postprocess_stdout.log") `
        2> (Join-Path $OutDir "postprocess_stderr.log")
    "postprocess exit code $LASTEXITCODE at $(Get-Date -Format o)" | Add-Content $RunLog

    & $Python -B diagnose_residuals.py --checkpoint $Latest --device cuda `
        --n-interior 4096 --n-interface 1024 --n-boundary 512 --n-ic 1024 `
        --output (Join-Path $PostDir "diagnostics_latest.txt") `
        > (Join-Path $OutDir "diagnose_stdout.log") `
        2> (Join-Path $OutDir "diagnose_stderr.log")
    "diagnose exit code $LASTEXITCODE at $(Get-Date -Format o)" | Add-Content $RunLog
} else {
    "no latest checkpoint found; skipped postprocess/diagnostics" | Add-Content $RunLog
}

"fresh 2h run finished $(Get-Date -Format o)" | Add-Content $RunLog
