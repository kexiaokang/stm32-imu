@echo off
chcp 65001 >nul
set ZIG=F:\anaconda\Lib\site-packages\ziglang\zig.exe

if not exist "%ZIG%" (
    echo [错误] 找不到编译器: %ZIG%
    echo        重新安装: pip install ziglang -i https://pypi.tuna.tsinghua.edu.cn/simple
    exit /b 1
)

"%ZIG%" cc main.c predict.c model_data.c -O2 -Wall -Wno-unused-parameter -o predict.exe
if errorlevel 1 (
    echo.
    echo [编译失败] 从上面第一条 error 看起。
    exit /b 1
)

echo.
echo [编译成功] 下面是运行结果：
echo ==============================================
predict.exe
echo ==============================================
