# Chat Endpoint Benchmark Script
# Tests performance of /chat endpoint with available models

# Configuration: Specify which models to benchmark (empty array = all models)
# Example: @("gpt2", "Qwen/Qwen3-0.6B")
$modelsToTest = @(
   
)

$serverUrl = "http://localhost:8005"
$testMessages = @(
    @{role="user"; content="Hello, how are you?"}
)

Write-Host "=== AI Server Chat Endpoint Benchmark ===" -ForegroundColor Cyan
Write-Host ""

# 0. Auto-pull models from HF_HOME/hub
$hfHome = $env:HF_HOME
if (-not $hfHome) {
    $hfHome = "E:\private-ai-server\models"  # fallback, adjust as needed
}
$hubPath = Join-Path $hfHome "hub"
Write-Host "0. Scanning for models in $hubPath..." -ForegroundColor Yellow
if (-not (Test-Path $hubPath)) {
    Write-Host "✗ HF_HOME hub path not found: $hubPath" -ForegroundColor Red
    exit 1
}
$modelFolders = Get-ChildItem -Path $hubPath -Directory | Where-Object { $_.Name -like 'models--*' }
if ($modelFolders.Count -eq 0) {
    Write-Host "✗ No model folders found in $hubPath" -ForegroundColor Red
    exit 1
}
$modelsToPull = @()
foreach ($folder in $modelFolders) {
    $name = $folder.Name -replace '^models--', '' -replace '--', '/'
    $modelsToPull += $name
}
Write-Host "Found $($modelsToPull.Count) model(s) to register: $($modelsToPull -join ', ')" -ForegroundColor Green

# Filter models to benchmark if specified
if ($modelsToTest.Count -gt 0) {
    $modelsToPull = $modelsToPull | Where-Object { $modelsToTest -contains $_ }
    Write-Host "Filtering to benchmark only: $($modelsToPull -join ', ')" -ForegroundColor Cyan
}

# Pull each model via /pull and wait for job completion
foreach ($modelName in $modelsToPull) {
    Write-Host "Registering model: $modelName via /pull..." -ForegroundColor Yellow
    $body = @{model=$modelName; init=$true} | ConvertTo-Json -Depth 5
    try {
        $pullResp = Invoke-RestMethod -Method POST -Uri "$serverUrl/pull" -ContentType "application/json" -Body $body
        $jobId = $pullResp.job_id
        Write-Host "  Job started: $jobId" -ForegroundColor Gray
        # Poll job status until succeeded or failed
        $maxWait = 600
        $waited = 0
        while ($true) {
            Start-Sleep -Seconds 2
            $waited += 2
            $job = Invoke-RestMethod -Uri "$serverUrl/jobs/$jobId"
            if ($job.status -eq "succeeded") {
                Write-Host "  Model $modelName registered successfully." -ForegroundColor Green
                break
            } elseif ($job.status -eq "failed") {
                Write-Host "  Model $modelName registration failed: $($job.error)" -ForegroundColor Red
                break
            } elseif ($waited -ge $maxWait) {
                Write-Host "  Timeout waiting for $modelName registration." -ForegroundColor Red
                break
            }
        }
    } catch {
        Write-Host "  Error registering ${modelName}: ${_}" -ForegroundColor Red
    }
}

# 1. Check server health
Write-Host "1. Checking server health..." -ForegroundColor Yellow
try {
    $health = Invoke-RestMethod -Uri "$serverUrl/health" -Method GET
    Write-Host "✓ Server is healthy" -ForegroundColor Green
    Write-Host "  - Uptime: $($health.uptime_seconds)s"
    Write-Host "  - Models cached: $($health.models_cached)"
    Write-Host "  - CUDA available: $($health.cuda_available)"
    Write-Host "  - GPU memory allocated: $($health.gpu_memory_allocated_mb) MB"
    Write-Host ""
} catch {
    Write-Host "✗ Server is not running at $serverUrl" -ForegroundColor Red
    Write-Host "  Please start the server: uvicorn app:app --host 0.0.0.0 --port 8005" -ForegroundColor Yellow
    exit 1
}

# 2. Get available models
Write-Host "2. Fetching available models..." -ForegroundColor Yellow
try {
    $modelsResponse = Invoke-RestMethod -Uri "$serverUrl/models" -Method GET
    $availableModels = $modelsResponse.models | Where-Object { $_.loaded -eq $true -or $_.local_path -ne $null }
    
    if ($availableModels.Count -eq 0) {
        Write-Host "✗ No models available for testing" -ForegroundColor Red
        Write-Host "  Please pull a model first:" -ForegroundColor Yellow
        Write-Host "  Invoke-RestMethod -Method POST -Uri $serverUrl/pull -ContentType 'application/json' -Body '{`"model`":`"gpt2`",`"init`":true}'" -ForegroundColor Yellow
        exit 1
    }
    
    Write-Host "✓ Found $($availableModels.Count) available model(s):" -ForegroundColor Green
    foreach ($model in $availableModels) {
        $sizeGB = if ($model.size_bytes) { [math]::Round($model.size_bytes / 1GB, 2) } else { "Unknown" }
        Write-Host "  - $($model.model) (${sizeGB} GB, backend: $($model.backend))"
    }
    Write-Host ""
} catch {
    Write-Host "✗ Failed to fetch models: $_" -ForegroundColor Red
    exit 1
}

# 3. Benchmark each model
$results = @()

foreach ($model in $availableModels) {
    $modelName = $model.model
    Write-Host "3. Benchmarking model: $modelName" -ForegroundColor Yellow
    Write-Host "   Configuration: max_tokens=100, temperature=0.7" -ForegroundColor Gray
    
    # Test parameters
    $maxTokens = 100
    $temperature = 0.7
    $numRuns = 3
    
    $runTimes = @()
    
    for ($i = 1; $i -le $numRuns; $i++) {
        Write-Host "   Run $i/$numRuns..." -NoNewline
        
        $body = @{
            model = $modelName
            messages = $testMessages
            max_tokens = $maxTokens
            temperature = $temperature
            stream = $false
        } | ConvertTo-Json -Depth 10
        
        try {
            $elapsed = Measure-Command {
                $response = Invoke-RestMethod -Method POST -Uri "$serverUrl/chat" -ContentType "application/json" -Body $body
            }
            
            $elapsedMs = [math]::Round($elapsed.TotalMilliseconds, 2)
            $runTimes += $elapsedMs
            
            # Extract metrics from response
            $totalDurationMs = if ($response.total_duration) { [math]::Round($response.total_duration / 1000000, 2) } else { 0 }
            $loadDurationMs = if ($response.load_duration) { [math]::Round($response.load_duration / 1000000, 2) } else { 0 }
            $evalDurationMs = if ($response.eval_duration) { [math]::Round($response.eval_duration / 1000000, 2) } else { 0 }
            
            Write-Host " ${elapsedMs}ms (server reported: ${totalDurationMs}ms)" -ForegroundColor Green
            
            if ($i -eq 1) {
                $previewText = if ($response.message.content.Length -gt 60) { 
                    $response.message.content.Substring(0, 60) 
                } else { 
                    $response.message.content 
                }
                Write-Host "      Response preview: ${previewText}..." -ForegroundColor Gray
                Write-Host "      Load duration: ${loadDurationMs}ms, Eval duration: ${evalDurationMs}ms" -ForegroundColor Gray
            }
            
        } catch {
            Write-Host " FAILED" -ForegroundColor Red
            Write-Host "      Error: $_" -ForegroundColor Red
            $runTimes += 0
        }
        
        # Small delay between runs
        if ($i -lt $numRuns) {
            Start-Sleep -Milliseconds 500
        }
    }
    
    # Calculate statistics
    $validRuns = $runTimes | Where-Object { $_ -gt 0 }
    if ($validRuns.Count -gt 0) {
        $avgTime = [math]::Round(($validRuns | Measure-Object -Average).Average, 2)
        $minTime = [math]::Round(($validRuns | Measure-Object -Minimum).Minimum, 2)
        $maxTime = [math]::Round(($validRuns | Measure-Object -Maximum).Maximum, 2)
        
        $results += [PSCustomObject]@{
            Model = $modelName
            Backend = $model.backend
            AvgTime = $avgTime
            MinTime = $minTime
            MaxTime = $maxTime
            SuccessRate = "$($validRuns.Count)/$numRuns"
        }
        
        Write-Host "   Summary: Avg=${avgTime}ms, Min=${minTime}ms, Max=${maxTime}ms" -ForegroundColor Cyan
    } else {
        Write-Host "   All runs failed for this model" -ForegroundColor Red
    }
    
    Write-Host ""
}

# 4. Display summary table
Write-Host "=== Benchmark Results Summary ===" -ForegroundColor Cyan
Write-Host ""

if ($results.Count -gt 0) {
    $results | Format-Table -AutoSize Model, Backend, @{Name="Avg (ms)"; Expression={$_.AvgTime}}, @{Name="Min (ms)"; Expression={$_.MinTime}}, @{Name="Max (ms)"; Expression={$_.MaxTime}}, SuccessRate
    
    # Find fastest model
    $fastest = $results | Sort-Object AvgTime | Select-Object -First 1
    Write-Host "🏆 Fastest model: $($fastest.Model) with avg $($fastest.AvgTime)ms" -ForegroundColor Green
} else {
    Write-Host "No successful benchmark results" -ForegroundColor Red
}

Write-Host ""
Write-Host "Benchmark complete!" -ForegroundColor Cyan
