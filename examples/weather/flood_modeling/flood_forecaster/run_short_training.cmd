@echo off
setlocal
powershell -ExecutionPolicy Bypass -File "%~dp0run_short_training.ps1" %*
exit /b %errorlevel%
