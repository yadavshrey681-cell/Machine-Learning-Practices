@echo off
echo ==================================================
echo   Antigravity Quick Sync to GitHub
echo ==================================================
cd /d "%~dp0"

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
