@echo off

call venv\Scripts\activate

set WINDOWS=5 10 15 20

for %%w in (%WINDOWS%) do (
    echo ======================================
    echo Training CNN for window size %%w
    echo ======================================

    python scripts\train_cnn.py --window_size %%w

    echo Done window size %%w
    echo.
)

echo All training completed.
pause