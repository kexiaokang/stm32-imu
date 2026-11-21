@echo off
chcp 65001 >nul
set ZIG=F:\anaconda\Lib\site-packages\ziglang\zig.exe

"%ZIG%" cc main_features.c features.c -O2 -Wall -Wno-unused-parameter -o features_test.exe
if errorlevel 1 (
    echo.
    echo [编译失败] 从上面第一条 error 看起。
    exit /b 1
)

echo.
echo [编译成功] 下面是运行结果：
echo ==============================================
features_test.exe
echo ==============================================
