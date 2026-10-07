<#
Register the Private AI server as a Windows background service (requires admin).

This uses `sc.exe` to create a service that starts the packaged server binary
(or a launcher) at boot. Requires an elevated PowerShell session.

Usage:
  .\scripts\register-service.ps1 -ServiceName PrivateAI-Server -Binary "D:\ai-server-py\dist\ai-server\ai-server.exe"
  .\scripts\register-service.ps1 -ServiceName PrivateAI-Server -Remove

Options:
  -ServiceName  Service name (default: PrivateAI-Server)
  -Binary       Full path to the server executable (required unless -Remove)
  -Remove       Delete the service instead of creating it
#>

param(
    [string]$ServiceName = "PrivateAI-Server",
    [string]$Binary = "",
    [switch]$Remove
)

$ErrorActionPreference = "Stop"

function Write-Log($msg) { Write-Host "[service] $msg" }

# Verify we have admin rights (sc.exe create requires elevation).
$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = New-Object Security.Principal.WindowsPrincipal($identity)
$isAdmin = $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $isAdmin) {
    Write-Error "This script requires an elevated (Administrator) PowerShell session to create a service."
    Write-Error "Re-run from an elevated terminal, or use register-autostart.ps1 (no admin needed)."
    exit 1
}

if ($Remove) {
    Write-Log "Stopping and removing service '$ServiceName'..."
    sc.exe stop $ServiceName 2>&1 | Out-Null
    sc.exe delete $ServiceName 2>&1 | Out-Null
    Write-Log "Done. Service removed."
    exit 0
}

if (-not $Binary) {
    Write-Error "Missing -Binary. Provide the full path to the server executable, or use -Remove."
    exit 1
}

if (-not (Test-Path $Binary)) {
    Write-Error "Binary not found: $Binary"
    exit 1
}

Write-Log "Creating service '$ServiceName' from $Binary ..."

# sc.exe create <name> binPath= "<binary>" start= auto DisplayName= "..."
sc.exe create $ServiceName binPath= "`"$Binary`"" start= auto DisplayName= "Private AI Inference Server"

if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to create service (exit $LASTEXITCODE)."
    exit 1
}

Write-Log "Starting service '$ServiceName'..."
sc.exe start $ServiceName 2>&1 | Out-Null

Write-Log "Service '$ServiceName' created and started. It will auto-start on boot."
