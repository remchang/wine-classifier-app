<#
============================================================
 xunlian.ps1 —— 训练并比较两种算法
 训练完成后会把关键指标打印出来，并把完整结果写进 models/xunlian_baogao.json
============================================================
#>
param(
    [int]$Zhongzi = 42,
    [string]$Shuchu = "models"
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

if (-not (Test-Path ".venv")) {
    Write-Host "x 还没建虚拟环境，先跑 .\scripts\anzhuang.ps1" -ForegroundColor Red
    exit 1
}

Write-Host "==> 开始训练（随机种子 = $Zhongzi）" -ForegroundColor Cyan
& ".\.venv\Scripts\python.exe" "src\xunlian.py" --zhongzi $Zhongzi --shuchu $Shuchu
if ($LASTEXITCODE -ne 0) { Write-Host "x 训练失败" -ForegroundColor Red; exit 1 }

Write-Host ""
Write-Host "产物：" -ForegroundColor Green
Get-ChildItem "$Shuchu" | ForEach-Object {
    Write-Host ("  {0,-22} {1,10:N0} 字节" -f $_.Name, $_.Length)
}
Write-Host ""
Write-Host "模型指纹（写进 model_meta.json，也抄一行到报告里）：" -ForegroundColor Cyan
$meta = Get-Content "$Shuchu\model_meta.json" -Raw -Encoding UTF8 | ConvertFrom-Json
Write-Host "  model.joblib SHA256 = $($meta.moxing_sha256)"
Write-Host "  数据指纹 SHA256     = $($meta.shuju_zhiwen_sha256)"
Write-Host "  选中算法            = $($meta.moxing_ming)"
