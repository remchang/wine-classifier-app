<#
============================================================
 qidong.ps1 —— 启动 Streamlit 推理界面

 启动前会检查模型文件在不在。不在就直接告诉你怎么训练，
 而不是让你打开一个报错的页面。
============================================================
#>
param(
    [int]$Duankou = 8502
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

if (-not (Test-Path "models\model.joblib")) {
    Write-Host "x 还没训练过模型。先执行：" -ForegroundColor Red
    Write-Host "    .\scripts\xunlian.ps1" -ForegroundColor Yellow
    exit 1
}

if (-not (Test-Path "models\model_meta.json")) {
    Write-Host "x 有 model.joblib 但没有 model_meta.json，模型不完整，请重新训练。" -ForegroundColor Red
    exit 1
}

Write-Host "==> 启动界面 http://localhost:$Duankou" -ForegroundColor Cyan
Write-Host "    （按 Ctrl+C 停止）" -ForegroundColor DarkGray
& ".\.venv\Scripts\python.exe" -m streamlit run app.py --server.port $Duankou --server.address 127.0.0.1
