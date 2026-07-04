param()

$ErrorActionPreference = "Stop"

Write-Host "Ngrok Live Demo Helper"
Write-Host ""

$ngrok = Get-Command ngrok -ErrorAction SilentlyContinue
if ($null -eq $ngrok) {
    Write-Host "ngrok was not found on PATH." -ForegroundColor Yellow
} else {
    Write-Host "ngrok found at: $($ngrok.Source)" -ForegroundColor Green
}

Write-Host ""
Write-Host "Manual steps:"
Write-Host "1. Start the backend locally:"
Write-Host "   cd backend"
Write-Host "   uvicorn main:app --host 0.0.0.0 --port 8000"
Write-Host ""
Write-Host "2. Start backend ngrok:"
Write-Host "   ngrok http 8000"
Write-Host ""
Write-Host "3. Copy the backend HTTPS forwarding URL."
Write-Host ""
Write-Host "4. Update frontend/.env.local with:"
Write-Host "   VITE_API_BASE_URL=<backend-ngrok-url>"
Write-Host ""
Write-Host "5. Start the frontend locally:"
Write-Host "   cd frontend"
Write-Host "   npm run dev -- --host 0.0.0.0"
Write-Host ""
Write-Host "6. Start frontend ngrok:"
Write-Host "   ngrok http 5173"
Write-Host ""
Write-Host "7. Share only the frontend HTTPS URL."
Write-Host ""
Write-Host "Reminder: keep backend, frontend, and both ngrok terminals running during the demo."

