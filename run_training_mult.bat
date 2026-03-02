@echo off

call venv\Scripts\activate

set WINDOWS=5 10 15 20

for %%w in (%WINDOWS%) do (
    @REM echo ======================================
    @REM echo Wavelet multilabel classification for window size %%w
    @REM echo ======================================

    @REM python scripts\ML1_wav_th.py --window_size %%w

    @REM echo Done window size %%w
    @REM echo.

    @REM echo ======================================
    @REM echo Wavelet multilabel LR classification for window size %%w
    @REM echo ======================================

    @REM python scripts\ML2_wav_LR.py --window_size %%w

    @REM echo Done window size %%w
    @REM echo.

    echo ======================================
    echo Wavelet multilabel LR classification masked for window size %%w
    echo ======================================

    python scripts\ML3_wav_LR_test_mask.py --window_size %%w

    echo Done window size %%w
    echo.

)

echo All training completed.
pause