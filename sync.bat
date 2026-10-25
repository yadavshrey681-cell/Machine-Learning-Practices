@echo off
echo ==================================================
echo   Antigravity Quick Sync to GitHub
echo ==================================================
cd /d "%~dp0"

where git >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    for /f "delims=" %%i in ('powershell -NoProfile -Command "(Get-ChildItem -Path \"$env:LOCALAPPDATA\GitHubDesktop\app-*\" | Sort-Object Name -Descending | Select-Object -First 1).FullName + '\resources\app\git\cmd'"') do (
        set "PATH=%%i;%PATH%"
    )
)

echo Staging all changes...
git add -A

echo Committing changes...
git commit -m "Manual sync: %date% %time%"

echo Pulling latest changes (rebase)...
git pull --rebase origin main

echo Pushing to GitHub...
git push origin main

echo.
if %ERRORLEVEL% EQU 0 (
    echo ==================================================
    echo   Sync completed successfully!
    echo ==================================================
) else (
    echo [ERROR] Push failed. If this is the first time, please make sure you are signed in to GitHub.
)
pause
