<#
============================================================
 anzhuang.ps1 —— 一次性环境安装

 做的事：
   1. 检查 python 版本
   2. 建虚拟环境 .venv
   3. 按 requirements.txt 装依赖（国内镜像，默认阿里云）
   4. 跑一遍测试确认环境没问题
============================================================
#>
param(
    [string]$Jingxiang = "https://mirrors.aliyun.com/pypi/simple/"
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

Write-Host "==> [1/4] 检查 python" -ForegroundColor Cyan
$py = Get-Command python -ErrorAction SilentlyContinue
if (-not $py) { Write-Host "  x 找不到 python，请先装 Python 3.11 或 3.12/3.13" -ForegroundColor Red; exit 1 }
python --version

if (-not (Test-Path ".venv")) {
    Write-Host "==> [2/4] 创建虚拟环境 .venv" -ForegroundColor Cyan
    python -m venv .venv
} else {
    Write-Host "==> [2/4] .venv 已存在，跳过创建" -ForegroundColor Cyan
}

Write-Host "==> [3/4] 安装依赖（镜像：$Jingxiang）" -ForegroundColor Cyan
& ".\.venv\Scripts\python.exe" -m pip install --upgrade pip -q -i $Jingxiang --trusted-host ([Uri]$Jingxiang).Host
& ".\.venv\Scripts\python.exe" -m pip install -r requirements.txt -i $Jingxiang --trusted-host ([Uri]$Jingxiang).Host
if ($LASTEXITCODE -ne 0) { Write-Host "  x 依赖安装失败" -ForegroundColor Red; exit 1 }

Write-Host "==> [4/4] 跑一遍测试自检" -ForegroundColor Cyan
& ".\.venv\Scripts\python.exe" -m pytest tests -q
Write-Host ""
Write-Host "装完了。下一步：" -ForegroundColor Green
Write-Host "  .\scripts\shuju-yangli.ps1   # 可选：看一眼数据长什么样"
Write-Host "  .\scripts\xunlian.ps1        # 训练模型"
Write-Host "  .\scripts\qidong.ps1         # 启动界面"
