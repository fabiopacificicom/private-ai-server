# Test script for Phase 1 Tasks 1.2 (Progress Tracking) and 1.4 (Timeout/Cancellation)
# Run after starting server: python -m uvicorn app:app --host 0.0.0.0 --port 8005

$baseUrl = "http://localhost:8005"

Write-Host "=== Phase 1 Testing: Progress Tracking & Timeout ===" -ForegroundColor Cyan
Write-Host ""

# Test 1.2: Progress Tracking
Write-Host "Test 1.2: Progress Tracking for Downloads" -ForegroundColor Yellow
Write-Host "Starting download of facebook/opt-125m (small model for quick test)..."

$pullBody = @{
    model = "facebook/opt-125m"
    init = $false
} | ConvertTo-Json

try {
    $pullResponse = Invoke-RestMethod -Method POST -Uri "$baseUrl/pull" -ContentType "application/json" -Body $pullBody
    $jobId = $pullResponse.job_id
    Write-Host "Job started: $jobId" -ForegroundColor Green
    
    # Poll job status and watch progress
    Write-Host "Polling progress (will show downloaded_bytes and progress)..."
    $maxPolls = 30
    $pollCount = 0
    
    while ($pollCount -lt $maxPolls) {
        Start-Sleep -Seconds 2
        $jobStatus = Invoke-RestMethod -Uri "$baseUrl/jobs/$jobId"
        
        $status = $jobStatus.status
        $progress = $jobStatus.progress
        $downloaded = $jobStatus.downloaded_bytes
        
        if ($downloaded -ne $null) {
            $downloadedMB = [math]::Round($downloaded / 1MB, 2)
            if ($progress -ne $null) {
                $progressPct = [math]::Round($progress * 100, 1)
                Write-Host "  Status: $status | Progress: $progressPct% | Downloaded: $downloadedMB MB" -ForegroundColor Cyan
            } else {
                Write-Host "  Status: $status | Progress: calculating... | Downloaded: $downloadedMB MB" -ForegroundColor Cyan
            }
        } else {
            Write-Host "  Status: $status | Progress: calculating..." -ForegroundColor Gray
        }
        
        if ($status -eq "succeeded" -or $status -eq "failed") {
            break
        }
        
        $pollCount++
    }
    
    if ($jobStatus.status -eq "succeeded") {
        Write-Host "✅ Download completed successfully!" -ForegroundColor Green
        if ($jobStatus.progress -ne $null) {
            Write-Host "   Final progress: $($jobStatus.progress * 100)%" -ForegroundColor Green
        } else {
            Write-Host "   Final progress: completed (size unknown)" -ForegroundColor Green
        }
        Write-Host "   Downloaded: $([math]::Round($jobStatus.downloaded_bytes / 1MB, 2)) MB" -ForegroundColor Green
    } else {
        Write-Host "❌ Download failed or timed out" -ForegroundColor Red
        Write-Host "   Error: $($jobStatus.error)" -ForegroundColor Red
    }
} catch {
    Write-Host "❌ Error during progress tracking test: $_" -ForegroundColor Red
}

Write-Host ""
Write-Host "Test 1.4: Request Timeout & Cancellation" -ForegroundColor Yellow

# First, ensure gpt2 is available for chat tests
Write-Host "Pre-requisite: Pulling gpt2 model for chat tests..."
$gpt2PullBody = @{
    model = "gpt2"
    init = $true
} | ConvertTo-Json

try {
    $gpt2PullResponse = Invoke-RestMethod -Method POST -Uri "$baseUrl/pull" -ContentType "application/json" -Body $gpt2PullBody
    $gpt2JobId = $gpt2PullResponse.job_id
    Write-Host "GPT2 pull job started: $gpt2JobId" -ForegroundColor Green
    
    # Wait for gpt2 pull to complete (or timeout after 120 seconds)
    $maxGpt2Polls = 60
    $gpt2PollCount = 0
    
    while ($gpt2PollCount -lt $maxGpt2Polls) {
        Start-Sleep -Seconds 2
        $gpt2JobStatus = Invoke-RestMethod -Uri "$baseUrl/jobs/$gpt2JobId"
        
        Write-Host "  GPT2 Status: $($gpt2JobStatus.status)" -ForegroundColor Gray
        
        if ($gpt2JobStatus.status -eq "succeeded") {
            Write-Host "✅ GPT2 model ready for chat tests!" -ForegroundColor Green
            break
        } elseif ($gpt2JobStatus.status -eq "failed") {
            Write-Host "❌ GPT2 pull failed: $($gpt2JobStatus.error)" -ForegroundColor Red
            Write-Host "Chat tests will likely fail" -ForegroundColor Yellow
            break
        }
        
        $gpt2PollCount++
    }
    
    if ($gpt2PollCount -ge $maxGpt2Polls) {
        Write-Host "⚠️  GPT2 pull timed out after 2 minutes" -ForegroundColor Yellow
        Write-Host "Chat tests may fail" -ForegroundColor Yellow
    }
} catch {
    Write-Host "⚠️  Warning: Failed to pull gpt2 model: $_" -ForegroundColor Yellow
    Write-Host "Chat tests may fail if gpt2 is not already cached" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Testing with very short timeout (should trigger timeout)..."

$chatBody = @{
    model = "gpt2"
    messages = @(
        @{role = "user"; content = "Write a very long story about artificial intelligence, at least 500 words."}
    )
    max_tokens = 1000
    temperature = 0.7
    timeout = 1  # 1 second - should timeout
} | ConvertTo-Json -Depth 10

try {
    Write-Host "Sending chat request with 1-second timeout..."
    $chatResponse = Invoke-RestMethod -Method POST -Uri "$baseUrl/chat" -ContentType "application/json" -Body $chatBody -ErrorAction Stop
    Write-Host "❌ Request should have timed out but didn't!" -ForegroundColor Red
} catch {
    $errorMessage = $_.Exception.Message
    if ($errorMessage -match "408" -or $errorMessage -match "timed out") {
        Write-Host "✅ Timeout triggered correctly (HTTP 408)" -ForegroundColor Green
        Write-Host "   Error message: $errorMessage" -ForegroundColor Gray
    } else {
        Write-Host "⚠️  Request failed but not with expected timeout error" -ForegroundColor Yellow
        Write-Host "   Error: $errorMessage" -ForegroundColor Gray
    }
}

Write-Host ""
Write-Host "Testing normal timeout (should succeed)..."

$chatBodyNormal = @{
    model = "gpt2"
    messages = @(
        @{role = "user"; content = "Say hello"}
    )
    max_tokens = 10
    temperature = 0.7
    timeout = 30  # 30 seconds - should succeed
} | ConvertTo-Json -Depth 10

try {
    $chatResponse = Invoke-RestMethod -Method POST -Uri "$baseUrl/chat" -ContentType "application/json" -Body $chatBodyNormal
    Write-Host "✅ Normal request completed successfully within timeout" -ForegroundColor Green
    Write-Host "   Response: $($chatResponse.message.content)" -ForegroundColor Gray
} catch {
    Write-Host "❌ Normal request failed: $_" -ForegroundColor Red
}

Write-Host ""
Write-Host "=== Testing Complete ===" -ForegroundColor Cyan
Write-Host "Summary:" -ForegroundColor White
Write-Host "  Task 1.2 (Progress Tracking): Check progress updates above" -ForegroundColor White
Write-Host "  Task 1.4 (Timeout/Cancellation): Check timeout behavior above" -ForegroundColor White
