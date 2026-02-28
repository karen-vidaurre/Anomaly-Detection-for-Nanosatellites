@echo off

call venv\Scripts\activate

set WINDOWS=5 10 15 20

for %%w in (%WINDOWS%) do (
    @REM echo ======================================
    @REM echo Variance based thresholding for window size %%w
    @REM echo ======================================

    @REM python scripts\D1_variance_th.py --window_size %%w

    @REM echo Done window size %%w
    @REM echo.

    echo ======================================
    echo Adaptative variance based thresholding for window size %%w
    echo ======================================

    python scripts\D2_adaptative_variance_th.py --window_size %%w

    echo Done window size %%w
    echo.

    echo ======================================
    echo Wavelet based thresholding for window size %%w
    echo ======================================

    python scripts\D3_wavelet_th.py --window_size %%w

    echo Done window size %%w
    echo.

    @REM echo ======================================
    @REM echo Training CNN for window size %%w
    @REM echo ======================================

    @REM python scripts\D4_train_cnn.py --window_size %%w

    @REM echo Done window size %%w
    @REM echo.
)

echo All training completed.
pause