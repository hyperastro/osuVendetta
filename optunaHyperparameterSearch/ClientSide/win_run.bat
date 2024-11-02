@echo off
:: Define Python embed zip and get-pip.py filenames and local install directory
set "PYTHON_ZIP=python-3.11.0-embed-amd64.zip"
set "GET_PIP=get-pip.py"
set "PYTHON_INSTALL_PATH=%~dp0Python311"

:: Check if Python is installed locally
if exist "%PYTHON_INSTALL_PATH%\python.exe" (
    echo Portable Python installation found.
    set "PYTHON_PATH=%PYTHON_INSTALL_PATH%\python.exe"
) else (
    echo Local Python installation not found. Setting up portable Python...
    
    :: Download Python embed zip if not present
    if exist "%PYTHON_ZIP%" (
        echo Python embed zip already exists.
    ) else (
        where curl >nul 2>&1
        IF ERRORLEVEL 1 (
            echo curl not found. Using PowerShell to download Python embed zip...
            powershell -Command "Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.11.0/%PYTHON_ZIP%' -OutFile '%PYTHON_ZIP%'"
        ) ELSE (
            curl -o "%PYTHON_ZIP%" "https://www.python.org/ftp/python/3.11.0/%PYTHON_ZIP%"
        )
    )
    
    :: Create the Python install directory if it doesn't exist
    if not exist "%PYTHON_INSTALL_PATH%" mkdir "%PYTHON_INSTALL_PATH%"
    
    :: Extract the embed zip to the Python install path
    echo Extracting Python to %PYTHON_INSTALL_PATH%...
    powershell -Command "Expand-Archive -Path '%PYTHON_ZIP%' -DestinationPath '%PYTHON_INSTALL_PATH%' -Force"
    
    :: Confirm extraction
    if not exist "%PYTHON_INSTALL_PATH%\python.exe" (
        echo Error: Python was not extracted to the expected directory.
        pause
        exit /b 1
    )
    
    :: Set PYTHON_PATH variable to the local Python installation
    set "PYTHON_PATH=%PYTHON_INSTALL_PATH%\python.exe"
    
    :: Clean up zip file
    del "%PYTHON_ZIP%"
)

:: Create directories for packages if needed
if not exist "%PYTHON_INSTALL_PATH%\Lib" mkdir "%PYTHON_INSTALL_PATH%\Lib"
if not exist "%PYTHON_INSTALL_PATH%\Lib\site-packages" mkdir "%PYTHON_INSTALL_PATH%\Lib\site-packages"
if not exist "%PYTHON_INSTALL_PATH%\DLLs" mkdir "%PYTHON_INSTALL_PATH%\DLLs"

:: Download full Python distribution for necessary files
echo Downloading full Python distribution...
set "PYTHON_FULL_ZIP=python-3.11.0-amd64.exe"
if not exist "%PYTHON_FULL_ZIP%" (
    powershell -Command "Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.11.0/%PYTHON_FULL_ZIP%' -OutFile '%PYTHON_FULL_ZIP%'"
)

:: Extract necessary files from full distribution
echo Extracting required files...
"%PYTHON_FULL_ZIP%" /quiet /layout "%TEMP%\python_full"
xcopy /E /I /Y "%TEMP%\python_full\Lib" "%PYTHON_INSTALL_PATH%\Lib"
xcopy /E /I /Y "%TEMP%\python_full\DLLs" "%PYTHON_INSTALL_PATH%\DLLs"

:: Clean up temporary files
rmdir /S /Q "%TEMP%\python_full"
del "%PYTHON_FULL_ZIP%"

:: Create python311._pth file with correct paths
echo Creating Python path configuration...
(
echo python311.zip
echo .
echo Lib\site-packages
echo .
echo import site
)> "%PYTHON_INSTALL_PATH%\python311._pth"

:: Set PYTHONPATH environment variable
set "PYTHONPATH=%PYTHON_INSTALL_PATH%\Lib;%PYTHON_INSTALL_PATH%\DLLs;%PYTHON_INSTALL_PATH%\Lib\site-packages"

:: Confirm Python path
echo Using local Python installation at: %PYTHON_PATH%

:: Download get-pip.py if it doesn't exist
if not exist "%GET_PIP%" (
    echo Downloading get-pip.py...
    powershell -Command "Invoke-WebRequest -Uri 'https://bootstrap.pypa.io/get-pip.py' -OutFile '%GET_PIP%'"
)

:: Install pip using get-pip.py
echo Installing pip...
"%PYTHON_PATH%" "%GET_PIP%"
if ERRORLEVEL 1 (
    echo Pip installation failed.
    pause
    exit /b 1
)

:: Remove get-pip.py after pip installation
del "%GET_PIP%"

:: Install dependencies using python -m pip
echo Installing dependencies from requirements.txt...
if exist requirements.txt (
    "%PYTHON_PATH%" -m pip install -r requirements.txt
    "%PYTHON_PATH%" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
) else (
    echo requirements.txt not found. Skipping dependency installation.
)

:: Run the main.py script if it exists
if exist main.py (
    echo Starting the program using local Python installation...
    start "" "%PYTHON_PATH%" main.py
) else (
    echo main.py not found.
    pause
)

:: Keep this window open
echo Installation completed. This window will remain open for reference.
pause
