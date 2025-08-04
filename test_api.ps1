<#
.SYNOPSIS
    Setup and test the HackRx Insurance Policy Analyzer API (Windows Version)
.DESCRIPTION
    This script will:
    1. Create a virtual environment
    2. Install all required dependencies
    3. Start the FastAPI server
    4. Run test requests to verify functionality
#>

# Configuration
$VENV_NAME = ".venv"
$REQUIREMENTS_FILE = "requirements.txt"
$API_URL = "http://localhost:8000"
$TEST_DOCUMENT_URL = "https://hackrx.blob.core.windows.net/assets/hackrx_6/policies/HDFHLIP23024V072223.pdf?sv=2023-01-03&st=2025-07-30T06%3A46%3A49Z&se=2025-09-01T06%3A46%3A00Z&sr=c&sp=rl&sig=9szykRKdGYj0BVm1skP%2BX8N9%2FRENEn2k7MQPUp33jyQ%3D"  # Replace with actual test document URL
$TEST_QUESTIONS = @(
    "What is the grace period for premium payment?",
    "Does this policy cover knee surgery?"
)

# Function to check command success
function Check-Success {
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Command failed with exit code $LASTEXITCODE"
        exit 1
    }
}

# 1. Create and activate virtual environment
Write-Host "`nCreating Python virtual environment..." -ForegroundColor Cyan
python -m venv $VENV_NAME
Check-Success

# Windows-specific activation
$activateScript = "$VENV_NAME\Scripts\Activate.ps1"
if (-not (Test-Path $activateScript)) {
    Write-Error "Virtual environment activation script not found at $activateScript"
    exit 1
}

# Dot-source the activation script
. $activateScript
Check-Success

# 2. Install dependencies
Write-Host "`nInstalling dependencies..." -ForegroundColor Cyan
pip install --upgrade pip
pip install -r $REQUIREMENTS_FILE
Check-Success

# 3. Start FastAPI server in background
Write-Host "`nStarting FastAPI server..." -ForegroundColor Cyan
$serverProcess = Start-Process -FilePath "python" -ArgumentList "-m uvicorn main:app --host 0.0.0.0 --port 8000" -PassThru

# Wait for server to start
Start-Sleep -Seconds 5

# 4. Test endpoints
Write-Host "`nTesting API endpoints..." -ForegroundColor Cyan

function Test-Endpoint {
    param (
        [string]$Url,
        [string]$Method = "Get",
        [object]$Body = $null,
        [hashtable]$Headers = @{}
    )
    
    try {
        $params = @{
            Uri = $Url
            Method = $Method
            ContentType = "application/json"
        }
        
        if ($Body) {
            $params.Body = $Body | ConvertTo-Json -Depth 5
        }
        
        if ($Headers) {
            $params.Headers = $Headers
        }
        
        $response = Invoke-RestMethod @params
        Write-Host "$Method $Url response:`n$($response | ConvertTo-Json -Depth 5)" -ForegroundColor Green
        return $true
    }
    catch {
        Write-Host "Error testing $Method $Url :" -ForegroundColor Red
        Write-Host $_.Exception.Message -ForegroundColor Red
        if ($_.Exception.Response) {
            $reader = New-Object System.IO.StreamReader($_.Exception.Response.GetResponseStream())
            $reader.BaseStream.Position = 0
            $reader.DiscardBufferedData()
            $responseBody = $reader.ReadToEnd()
            Write-Host "Response details: $responseBody" -ForegroundColor Red
        }
        return $false
    }
}

# Test endpoints
$rootTest = Test-Endpoint -Url "$API_URL/"
$healthTest = Test-Endpoint -Url "$API_URL/health"

$testBody = @{
    documents = $TEST_DOCUMENT_URL
    questions = $TEST_QUESTIONS
}

$apiTest = Test-Endpoint -Url "$API_URL/hackrx/run" -Method Post -Body $testBody -Headers @{
    "Authorization" = "Bearer 7294b64376d390e0c8800d2f7dd32943cbe143a7eeb1f7787d878ffb3d329995"
}

# 5. Clean up
Write-Host "`nStopping server..." -ForegroundColor Cyan
Stop-Process -Id $serverProcess.Id -Force

Write-Host "`nTest results:" -ForegroundColor Cyan
Write-Host "Root endpoint: $(if ($rootTest) {'Success'} else {'Failed'})" -ForegroundColor $(if ($rootTest) {'Green'} else {'Red'})
Write-Host "Health check: $(if ($healthTest) {'Success'} else {'Failed'})" -ForegroundColor $(if ($healthTest) {'Green'} else {'Red'})
Write-Host "API endpoint: $(if ($apiTest) {'Success'} else {'Failed'})" -ForegroundColor $(if ($apiTest) {'Green'} else {'Red'})

Write-Host "`nScript completed. Check above for any errors." -ForegroundColor Cyan