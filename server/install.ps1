Set-Location $PSScriptRoot
docker compose up -d --build
Write-Host "Server starting. Check status with: docker compose logs -f"
Write-Host "Once up, health check: curl.exe http://localhost:8000/health"
