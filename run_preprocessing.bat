@echo off

REM Activar el venv (MUY IMPORTANTE)
call venv\Scripts\activate

set WINDOWS=5 10 15 20

for %%w in (%WINDOWS%) do (
    echo ======================================
    echo Creating sliding windows: %%w
    echo ======================================

    @REM python scripts\create_sliding_windows.py ^
    @REM     --input_csv data\raw\sim_357_bal_t1.csv ^
    @REM     --output_dir data\processed\windows_%%w ^
    @REM     --window_size %%w ^
    @REM     --analyze

    echo Done window size %%w
    echo.

    echo ======================================
    echo Splitting dataset %%w
    echo ======================================
    python scripts\split_dataset.py ^
        --window_size %%w

    echo Done window size split %%w
    echo.
)

echo All preprocessing finished.
pause