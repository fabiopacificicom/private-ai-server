$base='http://127.0.0.1:8005'
Write-Host 'Polling /diag (up to 30s)'
$diag = $null
for ($i=0; $i -lt 15; $i++) {
  try {
    $diag = Invoke-RestMethod -Uri ("$base/diag") -UseBasicParsing -TimeoutSec 5
    Write-Host "DIAG:"; $diag | ConvertTo-Json -Depth 6
    break
  } catch {
    Write-Host "Not up yet... ($i)"; Start-Sleep -Seconds 2
  }
}
if (-not $diag) { Write-Host "Failed to reach /diag"; exit 1 }

# Trigger non-blocking pull for gpt2 (small model)
Write-Host "\n-> Triggering /pull for model 'gpt2' (init=true)"
$payload = @{ model = 'gpt2'; init = $true } | ConvertTo-Json
try {
  $pullRes = Invoke-RestMethod -Uri ("$base/pull") -Method Post -Body $payload -ContentType 'application/json' -TimeoutSec 30
  Write-Host "Pull accepted:"; $pullRes | ConvertTo-Json -Depth 6
} catch { Write-Host "Pull request failed: $_"; exit 1 }

$jobId = $pullRes.job_id
Write-Host "JobId: $jobId\nPolling job status (max 300s)"
$max=300; $elapsed=0
while ($true) {
  try { $j = Invoke-RestMethod -Uri ("$base/jobs/$jobId") -TimeoutSec 30 } catch { Write-Host "Failed to fetch job: $_"; break }
  Write-Host ("JOB: " + $j.status + "  notes:" + ($j.notes -as [string]) + "  error:" + ($j.error -as [string]))
  if ($j.status -ne 'running' -and $j.status -ne 'queued') { break }
  Start-Sleep -Seconds 3
  $elapsed += 3
  if ($elapsed -gt $max) { Write-Host 'Polling timeout'; break }
}
Write-Host "Final job state:"; $j | ConvertTo-Json -Depth 10

if ($j.status -eq 'succeeded') {
  Write-Host "\n-> GET /models"
  try { $models = Invoke-RestMethod -Uri ("$base/models") -TimeoutSec 30; $models | ConvertTo-Json -Depth 6 } catch { Write-Host "Failed to list models: $_" }

  Write-Host "\n-> POST /chat smoke test (gpt2)"
  $chatBody = @{ model = 'gpt2'; messages = @(@{ role = 'user'; content = 'Hello, please respond with a short confirmation that you are running.' }) } | ConvertTo-Json
  try {
    $chatRes = Invoke-RestMethod -Uri ("$base/chat") -Method Post -Body $chatBody -ContentType 'application/json' -TimeoutSec 120
    Write-Host "Chat response:"; $chatRes | ConvertTo-Json -Depth 8
  } catch { Write-Host "Chat failed: $_" }
} else {
  Write-Host "Job did not succeed; skipping chat test"
}

Write-Host "\n-> Done. To stop the server job: Stop-Job -Name ai-server; Remove-Job -Name ai-server"