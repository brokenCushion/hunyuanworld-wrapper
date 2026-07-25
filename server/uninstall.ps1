# Clean uninstall: removes containers, network, volumes, and the built image.
# Nothing outside this project folder is touched. Model weights and job outputs
# live in .\data\ -- delete that folder (or this whole directory) for a full
# wipe, or keep .\data\hf-cache to skip re-downloading weights.
Set-Location $PSScriptRoot
docker compose down -v --rmi all
Write-Host "Containers, volumes, and image removed."
Write-Host "Weights/job data still on disk at .\data\ -- delete manually for a full wipe."
