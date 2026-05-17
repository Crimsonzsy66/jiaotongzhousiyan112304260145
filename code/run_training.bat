@echo off
chcp 65001 > nul
echo ==========================================
echo 开始运行优化训练 - YOLOv8s
echo ==========================================
echo.

cd /d "d:\Trae programes\交通标志检测\第4次实验数据及提交格式"

python "d:\Trae programes\交通标志检测\第4次实验数据及提交格式\code\train_model.py"

if %errorlevel% equ 0 (
    echo.
    echo ==========================================
    echo 训练完成！开始生成提交文件...
    echo ==========================================
    echo.
    
    python "d:\Trae programes\交通标志检测\第4次实验数据及提交格式\code\run_infer.py"
    
    if %errorlevel% equ 0 (
        echo.
        echo ==========================================
        echo 完成！提交文件已保存
        echo ==========================================
    ) else (
        echo.
        echo 推理过程中出现错误
    )
) else (
    echo.
    echo 训练过程中出现错误
)

pause
