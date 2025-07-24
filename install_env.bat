@echo off
REM Create a Python virtual environment
python -m venv venv

REM Activate the virtual environment
call .\venv\Scripts\Activate.ps1 

pip install -r requirements.txt

REM Install the package in editable mode
pip install -e .

echo Installation complete. Virtual environment is activated.
