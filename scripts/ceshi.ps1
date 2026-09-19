<#
============================================================
 ceshi.ps1 —— 一键跑全部测试

 三类：
   1. pytest（数据/划分、Pipeline、推理与输入校验）
   2. 训练可复现性（同种子跑两次比对指标）
   3. 模型元数据自检（SHA256、特征顺序）
============================================================
#>
$ErrorActionPreference = "Continue"
Set-Location (Split-Path $PSScriptRoot -Parent)

Write-Host "===== 1. pytest 自动化测试 =====" -ForegroundColor Cyan
& ".\.venv\Scripts\python.exe" -m pytest tests -q
$pytestMa = $LASTEXITCODE

Write-Host ""
Write-Host "===== 2. 训练可复现性（同种子两次）=====" -ForegroundColor Cyan
$t1 = Join-Path $env:TEMP "wine_fuyan_1"
$t2 = Join-Path $env:TEMP "wine_fuyan_2"
Remove-Item -Recurse -Force $t1, $t2 -ErrorAction SilentlyContinue

& ".\.venv\Scripts\python.exe" "src\xunlian.py" --zhongzi 42 --shuchu $t1 --baogao "$t1\b.json" | Out-Null
& ".\.venv\Scripts\python.exe" "src\xunlian.py" --zhongzi 42 --shuchu $t2 --baogao "$t2\b.json" | Out-Null

$a = Get-Content "$t1\b.json" -Raw -Encoding UTF8 | ConvertFrom-Json
$b = Get-Content "$t2\b.json" -Raw -Encoding UTF8 | ConvertFrom-Json

$yizhi = ($a.ceshi_ji.f1_macro -eq $b.ceshi_ji.f1_macro) -and
         ($a.ceshi_ji.accuracy -eq $b.ceshi_ji.accuracy) -and
         ($a.xuanzhong -eq $b.xuanzhong)
if ($yizhi) {
    Write-Host "  [PASS] 两次训练结果完全一致：acc=$($a.ceshi_ji.accuracy)  f1_macro=$($a.ceshi_ji.f1_macro)  选中=$($a.xuanzhong)" -ForegroundColor Green
} else {
    Write-Host "  [FAIL] 两次训练结果不一致，可复现性有问题！" -ForegroundColor Red
}

Write-Host ""
Write-Host "===== 3. 模型元数据自检 =====" -ForegroundColor Cyan
if (Test-Path "models\model_meta.json") {
    $meta = Get-Content "models\model_meta.json" -Raw -Encoding UTF8 | ConvertFrom-Json
    $zhen = (Get-FileHash "models\model.joblib" -Algorithm SHA256).Hash.ToLower()
    if ($zhen -eq $meta.moxing_sha256) {
        Write-Host "  [PASS] model.joblib 的 SHA256 与元数据一致" -ForegroundColor Green
    } else {
        Write-Host "  [FAIL] SHA256 对不上：文件 $zhen / 元数据 $($meta.moxing_sha256)" -ForegroundColor Red
    }
    Write-Host "         特征数 = $($meta.tezheng_ming.Count)   类别数 = $($meta.leibie_ming.Count)"
    Write-Host "         sklearn = $($meta.huanjing.scikit_learn)   commit = $($meta.git_commit)"
} else {
    Write-Host "  ! models\model_meta.json 不存在，先跑 .\scripts\xunlian.ps1" -ForegroundColor Yellow
}

Remove-Item -Recurse -Force $t1, $t2 -ErrorAction SilentlyContinue
Write-Host ""
if ($pytestMa -eq 0) { Write-Host "全部完成。" -ForegroundColor Green } else { Write-Host "pytest 有失败项，看上面的输出。" -ForegroundColor Red }
