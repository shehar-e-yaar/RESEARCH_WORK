# PowerShell Pipeline Runner for Kafka & Spark Structured Streaming
Write-Host "==========================================================================" -ForegroundColor Cyan
Write-Host "   REAL-TIME DDOS DETECTION PIPELINE (KAFKA + SPARK STRUCTURED STREAMING)" -ForegroundColor Cyan
Write-Host "==========================================================================" -ForegroundColor Cyan

Write-Host "`nSelect Execution Mode:"
Write-Host "  [1] Native File-Stream Mode (No Docker Required - Ready Now)"
Write-Host "  [2] Real Kafka Broker Mode (Requires Docker Desktop Running)"
$choice = Read-Host "Enter option (1 or 2, default is 1)"

if ($choice -eq "2") {
    Write-Host "`n[Step 1] Checking Docker daemon status..." -ForegroundColor Yellow
    docker info > $null 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Docker Desktop is not running. Please start Docker Desktop first!" -ForegroundColor Red
        Exit 1
    }

    Write-Host "[Step 2] Starting Apache Kafka container (KRaft mode)..." -ForegroundColor Yellow
    docker compose up -d
    Start-Sleep -Seconds 5

    Write-Host "[Step 3] Launching Spark Structured Streaming Consumer..." -ForegroundColor Yellow
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "python spark_streaming_consumer.py --mode kafka --count 10000"

    Start-Sleep -Seconds 8

    Write-Host "[Step 4] Launching Real-Time Kafka Flow Producer..." -ForegroundColor Yellow
    python kafka_producer.py --rate 2500 --count 10000
} else {
    Write-Host "`nRunning Spark Structured Streaming evaluation..." -ForegroundColor Green
    python spark_streaming_consumer.py --mode file --count 6000
}

Write-Host "`n[Complete] Check results/kafka_spark_streaming_evaluation.json for output!" -ForegroundColor Green
