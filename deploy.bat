@echo off
setlocal

REM ========================================
REM MonkeySee Deployment Script
REM Deploy code to OrangePi vision computer
REM ========================================

REM Configuration
set SOURCE_DIR=X:\Vision\MonkeySee
set TARGET_USER=orangepi
set TARGET_HOST=10.8.46.204
set TARGET_PORT=22
set TARGET_DIR=/home/orangepi/MonkeySee

REM Parse command line arguments
if "%1"=="1" set TARGET_HOST=funkyvision1
if "%1"=="2" set TARGET_HOST=funkyvision2
if "%1"=="3" set TARGET_HOST=funkyvision3
if "%1"=="4" set TARGET_HOST=funkyvision4

echo ========================================
echo Deploying MonkeySee to %TARGET_HOST%
echo ========================================
echo Source: %SOURCE_DIR%
echo Target: %TARGET_USER%@%TARGET_HOST%:%TARGET_DIR%
echo Port: %TARGET_PORT%
echo.

REM Create temporary directory for filtered files
set TEMP_DIR=%TEMP%\MonkeySee_deploy_%RANDOM%
echo Creating temporary deployment directory...
mkdir "%TEMP_DIR%"

REM Copy files excluding unnecessary items
echo Copying files (excluding __pycache__, .git, etc.)...
robocopy "%SOURCE_DIR%" "%TEMP_DIR%" /E /XD __pycache__ .git .vscode /XF *.pyc *.pyo .gitignore deploy.bat test_*.py

REM Deploy via SCP
echo.
echo Deploying to OrangePi...
scp -P %TARGET_PORT% -r "%TEMP_DIR%\*" "%TARGET_USER%@%TARGET_HOST%:%TARGET_DIR%/"

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================
    echo Deployment successful!
    echo ========================================
    echo.
    echo The system will need to be restarted on the OrangePi.
    echo To restart, run:
    echo   ssh -p %TARGET_PORT% %TARGET_USER%@%TARGET_HOST%
    echo   sudo systemctl restart monkeysee
) else (
    echo.
    echo ========================================
    echo Deployment failed!
    echo ========================================
    echo Check your network connection and credentials.
)

REM Cleanup
echo.
echo Cleaning up temporary files...
rmdir /S /Q "%TEMP_DIR%"

echo.
echo Process completed
endlocal
pause

