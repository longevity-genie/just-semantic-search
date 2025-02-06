@echo off
setlocal EnableDelayedExpansion

:: Default values
set "OUTPUT_DIR=parsed"
set "LANG=en"
set "MD_DIR="
set "RUN_AS_ROOT="

:: Parse command line arguments
:parse_args
if "%~1"=="" goto check_input
if "%~1"=="-i" set "INPUT_DIR=%~2" & shift & shift & goto parse_args
if "%~1"=="-o" set "OUTPUT_DIR=%~2" & shift & shift & goto parse_args
if "%~1"=="-l" set "LANG=%~2" & shift & shift & goto parse_args
if "%~1"=="-m" set "MD_DIR=%~2" & shift & shift & goto parse_args
if "%~1"=="-r" set "RUN_AS_ROOT=true" & shift & goto parse_args
if "%~1"=="-h" goto show_help
echo Invalid option: %~1
exit /b 1

:show_help
echo Usage: %~nx0 -i input_dir [-o output_dir] [-l language] [-m markdown_dir] [-r]
echo Options:
echo   -i: Input directory (required)
echo   -o: Output directory (default: parsed)
echo   -l: Language (default: en)
echo   -m: Markdown directory (optional, to collect all .md files)
echo   -r: Run as root (optional, default: run as current user)
exit /b 0

:check_input
if not defined INPUT_DIR (
    echo MinerU PDF Parser
    echo ----------------
    echo This script uses the MinerU docker container to parse a folder containing PDF files.
    echo MinerU container is huge, it may pull 14GB of data to your machine.
    echo It extracts text and metadata from PDFs using GPU acceleration.
    echo.
    echo Usage: %~nx0 -i input_dir [-o output_dir] [-l language] [-m markdown_dir] [-r]
    echo.
    echo Options:
    echo   -i: Input directory containing PDF files (required)
    echo   -o: Output directory for parsed results (default: parsed)
    echo   -l: Language of the PDFs (default: en)
    echo   -m: Markdown directory (optional, to collect all .md files)
    echo   -r: Run as root (optional, default: run as current user)
    echo.
    echo Example:
    echo   %~nx0 -i ./pdfs -o ./results -l en -m ./markdown
    exit /b 1
)

:: Convert to absolute paths
for %%i in ("%INPUT_DIR%") do set "INPUT_ABS=%%~fi"
for %%i in ("%OUTPUT_DIR%") do set "OUTPUT_ABS=%%~fi"

:: Create output directory if it doesn't exist
if not exist "%OUTPUT_ABS%" (
    mkdir "%OUTPUT_ABS%"
    echo Created output directory: %OUTPUT_ABS%
)

:: Handle markdown directory if specified
if defined MD_DIR (
    for %%i in ("%MD_DIR%") do set "MD_ABS=%%~fi"
    if not exist "!MD_ABS!" (
        mkdir "!MD_ABS!"
        echo Created markdown directory: !MD_ABS!
    )
    set "MD_MOUNT=-v !MD_ABS!:/data/markdown"
    set "COPY_CMD=&& find /data/output -name '*.md' -exec cp {} /data/markdown/ \;"
) else (
    set "MD_MOUNT="
    set "COPY_CMD="
)

:: Set user parameters for docker
if defined RUN_AS_ROOT (
    set "USER_PARAMS="
) else (
    set "USER_PARAMS=--user 1000:1000"
)

:: Run the docker command
docker run --gpus=all ^
    %USER_PARAMS% ^
    -v "%INPUT_ABS%:/data/input" ^
    -v "%OUTPUT_ABS%:/data/output" ^
    %MD_MOUNT% ^
    raychanan/mineru /bin/bash -c "magic-pdf ^
    -p '/data/input' ^
    -o '/data/output' ^
    -l '%LANG%' ^
    %COPY_CMD%"

endlocal
