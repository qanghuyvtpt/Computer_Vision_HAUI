# Chay ca 2 service: Backend AI (8000) + Frontend web (5266)
# Cach dung: click phai -> Run with PowerShell, hoac: .\start_app.ps1

$Root = $PSScriptRoot
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")

if (-not (Test-Path "$Root\frontend\.venv\Scripts\uvicorn.exe")) {
    Write-Host "Chua co venv. Chay: python -m venv frontend\.venv" -ForegroundColor Red
    exit 1
}
if (-not (Test-Path "$Root\train_model\best_animal_classifier.pt")) {
    Write-Host "Thieu file model: train_model\best_animal_classifier.pt" -ForegroundColor Yellow
}

Write-Host "Khoi dong Backend AI  -> http://127.0.0.1:8000/health" -ForegroundColor Cyan
Start-Process powershell -ArgumentList @(
    "-NoExit", "-Command",
    "Set-Location '$Root'; & '$Root\frontend\.venv\Scripts\uvicorn.exe' backend.app:app --host 127.0.0.1 --port 8000"
)

Start-Sleep -Seconds 3

Write-Host "Khoi dong Frontend web -> http://127.0.0.1:5266" -ForegroundColor Green
Start-Process powershell -ArgumentList @(
    "-NoExit", "-Command",
    "`$env:ASPNETCORE_ENVIRONMENT='Development'; Set-Location '$Root\frontend'; dotnet run --urls http://127.0.0.1:5266"
)

Write-Host "Doi backend san sang..." -ForegroundColor Yellow
$ok = $false
for ($i = 0; $i -lt 30; $i++) {
    try {
        $h = Invoke-RestMethod -Uri "http://127.0.0.1:8000/health" -TimeoutSec 2
        if ($h.status -eq "ok") { $ok = $true; break }
    } catch {}
    Start-Sleep -Seconds 2
}
if (-not $ok) {
    Write-Host "Backend chua san sang. Doi them roi mo http://127.0.0.1:5266" -ForegroundColor Yellow
} else {
    Write-Host "Backend OK. checkpointExists=$($h.checkpointExists)" -ForegroundColor Cyan
}

Start-Sleep -Seconds 3
Start-Process "http://127.0.0.1:5266"
Write-Host ""
Write-Host "=== WEB HOAN CHINH ===" -ForegroundColor Green
Write-Host "  Website (upload anh):  http://127.0.0.1:5266"
Write-Host "  API AI (chi kiem tra): http://127.0.0.1:8000/health"
Write-Host "  Giu 2 cua so terminal mo. Upload anh -> Phan Tich Ngay."
