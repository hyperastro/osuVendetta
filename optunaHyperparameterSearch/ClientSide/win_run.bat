@echo off
:: Define the Python installer filename and target install directory
set "PYTHON_INSTALLER=python-3.11.0-amd64.exe"
set "PYTHON_INSTALL_PATH=C:\Python311"

:: Check if Python is installed and avoid the WindowsApps alias
for /f "tokens=*" %%i in ('where python 2^>nul') do (
    set "PYTHON_PATH=%%i"
    if /i "%%~dpi"=="C:\Users\Gui\AppData\Local\Microsoft\WindowsApps\" (
        set "PYTHON_PATH="
    )
)

:: If PYTHON_PATH is not defined, proceed with installation
IF NOT DEFINED PYTHON_PATH (
    echo Valid Python installation not found. Installing Python...
    :: Check if the Python installer already exists
    if exist "%PYTHON_INSTALLER%" (
        echo Python installer already exists.
    ) else (
        :: Try downloading with curl or fallback to PowerShell if curl is unavailable
        where curl >nul 2>&1
        IF ERRORLEVEL 1 (
            echo curl not found. Attempting to use PowerShell to download Python installer...
            powershell -Command "Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.11.0/%PYTHON_INSTALLER%' -OutFile '%PYTHON_INSTALLER%'"
        ) ELSE (
            curl -o "%PYTHON_INSTALLER%" "https://www.python.org/ftp/python/3.11.0/%PYTHON_INSTALLER%"
        )
    )
    :: Pause to view any download errors
    pause
    
    :: Install Python silently with correct syntax
    "%PYTHON_INSTALLER%" /quiet InstallAllUsers=1 PrependPath=1 TargetDir="%PYTHON_INSTALL_PATH%"
    if ERRORLEVEL 1 (
        echo Failed to install Python. Please install it manually.
        pause
        exit /b 1
    )
    
    :: Wait a moment for installation to complete
    timeout /t 5 /nobreak
    
    :: Set PYTHON_PATH to the installed Python executable
    set "PYTHON_PATH=%PYTHON_INSTALL_PATH%\python.exe"
    
    :: Clean up installer
    del "%PYTHON_INSTALLER%"
    
    :: Refresh PATH environment variable
    call refreshenv.cmd 2>nul || (
        :: If refreshenv is not available, update PATH manually
        setx PATH "%PYTHON_INSTALL_PATH%;%PYTHON_INSTALL_PATH%\Scripts;%PATH%"
        set "PATH=%PYTHON_INSTALL_PATH%;%PYTHON_INSTALL_PATH%\Scripts;%PATH%"
    )
) ELSE (
    echo Valid Python installation found at %PYTHON_PATH%.
)

:: Wait for PATH to update
timeout /t 2 /nobreak

:: Verify Python installation
"%PYTHON_PATH%" --version
if ERRORLEVEL 1 (
    echo Failed to verify Python installation.
    pause
    exit /b 1
)

:: Check if pip is installed and upgrade it
"%PYTHON_PATH%" -m pip --version >nul 2>&1
IF ERRORLEVEL 1 (
    echo Pip not found. Installing pip...
    "%PYTHON_PATH%" -m ensurepip --upgrade
)

:: Upgrade pip to latest version
echo Upgrading pip to latest version...
"%PYTHON_PATH%" -m pip install --upgrade pip

:: Install dependencies from requirements.txt if it exists
IF EXIST requirements.txt (
    echo Installing dependencies from requirements.txt...
    "%PYTHON_PATH%" -m pip install -r requirements.txt
    "%PYTHON_PATH%" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
) ELSE (
    echo requirements.txt not found. Skipping dependency installation.
)

:: Create a new command window to run the program
if exist main.py (
    echo Starting the program...
    start cmd /k ""%PYTHON_PATH%" main.py"
) else (
    echo main.py not found.
    pause
)

:: Keep this window open
echo Installation completed. This window will remain open for reference.
pause