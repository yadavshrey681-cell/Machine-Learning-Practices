# Auto Git Sync Script for Machine Learning Workspace
# Automatically commits and pushes changes to GitHub periodically or on detection

param(
    [int]$IntervalSeconds = 30
)

$repoPath = (Resolve-Path "$PSScriptRoot\..").Path
Set-Location -Path $repoPath

# Ensure git is available in PATH
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    $latestApp = Get-ChildItem -Path "$env:LOCALAPPDATA\GitHubDesktop\app-*" -ErrorAction SilentlyContinue | Sort-Object Name -Descending | Select-Object -First 1
    if ($latestApp) {
        $gitCmd = Join-Path $latestApp.FullName "resources\app\git\cmd"
        if (Test-Path "$gitCmd\git.exe") {
            $env:PATH = "$gitCmd;" + $env:PATH
        }
    }
}

Write-Host "==================================================" -ForegroundColor Cyan
Write-Host " Antigravity Auto-Sync to GitHub Started          " -ForegroundColor Green
Write-Host " Repository : $repoPath" -ForegroundColor Gray
Write-Host " Interval   : Every $IntervalSeconds seconds" -ForegroundColor Gray
Write-Host " Target     : origin/main" -ForegroundColor Gray
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "Monitoring for changes... (Press Ctrl+C to stop)`n" -ForegroundColor DarkGray

while ($true) {
    try {
        # Check if there are untracked files or uncommitted modifications
        $changes = git status --porcelain 2>$null
        $ahead = git status -sb 2>$null | Select-String "ahead"

        if ($changes -or $ahead) {
            $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
            Write-Host "[$timestamp] Changes detected in workspace!" -ForegroundColor Yellow

            # Stage all changes
            git add -A

            # Commit if changes exist in staging area
            $stagedChanges = git diff --cached --name-only
            if ($stagedChanges) {
                $fileCount = ($stagedChanges | Measure-Object).Count
                $commitMsg = "Auto-sync: update $fileCount files [$timestamp]"
                git commit -m $commitMsg | Out-Null
                Write-Host "[$timestamp] Committed: $commitMsg" -ForegroundColor Cyan
            }

            # Pull latest changes with rebase before pushing to avoid conflicts
            git pull --rebase origin main 2>$null | Out-Null

            # Push to GitHub
            $pushOutput = git push origin main 2>&1
            if ($LASTEXITCODE -eq 0) {
                Write-Host "[$timestamp] Successfully synced and pushed to GitHub!" -ForegroundColor Green
            } else {
                Write-Warning "[$timestamp] Push failed: $pushOutput"
            }
        }
    }
    catch {
        Write-Warning "Error during auto-sync cycle: $_"
    }

    Start-Sleep -Seconds $IntervalSeconds
}
