@echo off
chcp 65001 >nul
set ZIG=F:\anaconda\Lib\site-packages\ziglang\zig.exe

"%ZIG%" cc main_full.c predict.c features.c model_data.c -O2 -Wall -Wno-unused-parameter -o full_test.exe
if errorlevel 1 (
    echo.
    echo [编译失败] 从上面第一条 error 看起。
    exit /b 1
)

echo.
echo [编译成功] 下面是运行结果（451 个窗口要算几秒，等着）：
echo ==============================================
full_test.exe
echo ==============================================
