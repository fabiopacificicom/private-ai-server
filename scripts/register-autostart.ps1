<#
Register a logon scheduled task so Private AI services start automatically
when the user logs in. Requires NO admin rights (runs at logon, limited).

Usage:
  .\scripts\register-autostart.ps1 -TaskName PrivateAI-Services -Command "cmd /c start /min D:\ai-server-py\start-services.cmd"
  .\scripts\register-autostart.ps1 -TaskName PrivateAI-Services -Remove

Options:
  -TaskName   Name of the scheduled task (default: PrivateAI-Services)
  -Command    Full command to run at logon (required unless -Remove)
  -Remove     Delete the task instead of creating it
#>

param(
    [string]$TaskName = "PrivateAI-Services",
    [string]$Command = "",
    [switch]$Remove
)

$ErrorActionPreference = "Stop"

function Write-Log($msg) { Write-Host "[autostart] $msg" }

if ($Remove) {
    Write-Log "Removing scheduled task '$TaskName'..."
    schtasks /Delete /TN $TaskName /F 2>&1 | Out-Null
    Write-Log "Done. Task removed."
    exit 0
}

if (-not $Command) {
    Write-Error "Missing -Command. Provide the command to run at logon, or use -Remove."
    exit 1
}

Write-Log "Registering logon task '$TaskName'..."
Write-Log "Command: $Command"

# /SC ONLOGON runs at user logon; /RL LIMITED needs no admin; /F overwrites.
schtasks /Create /TN $TaskName /TR $Command /SC ONLOGON /RL LIMITED /F

if ($LASTEXITCODE -eq 0) {
    Write-Log "Scheduled task '$TaskName' registered successfully."
    Write-Log "It will run automatically at your next logon."
} else {
    Write-Error "Failed to register scheduled task (exit $LASTEXITCODE)."
    exit 1
}
