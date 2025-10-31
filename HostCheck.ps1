<#
.SYNOPSIS
  Checks Windows Host prerequisites relevant to WSL2 GPU Passthrough for ROCm.
.DESCRIPTION
  This script gathers information about Windows version, WSL status, AMD GPU driver (best effort),
  and provides reminders for manual checks needed for Docker Desktop and precise driver version.
.NOTES
  Run this script in PowerShell as Administrator for best results.
  Driver version reported by WMI might not exactly match the Adrenalin version number. Manual check is recommended.
#>

# ==================================================
# WINDOWS HOST PRE-CHECK REPORT (PowerShell)
# Generated: $(Get-Date)
# ==================================================

# --- Windows Version ---
Write-Host "`n[WINDOWS INFORMATION]"
try {
    $OsInfo = Get-ComputerInfo -Property OsName, OsVersion, OsBuildNumber
    Write-Host "OS Name: $($OsInfo.OsName)"
    Write-Host "OS Version: $($OsInfo.OsVersion)"
    Write-Host "OS Build: $($OsInfo.OsBuildNumber)"
} catch {
    Write-Host "OS Info (via WMI): $((Get-WmiObject -Class Win32_OperatingSystem).Caption) / Version: $((Get-WmiObject -Class Win32_OperatingSystem).Version)"
    Write-Host "Error getting full ComputerInfo (may need newer PowerShell or run as Admin)."
}

# --- WSL Status ---
Write-Host "`n[WSL STATUS]"
try {
    Write-Host "--- wsl --version ---"
    wsl --version | Out-String | Select-Object -SkipLast 1 # Skip potential empty line
    Write-Host "--- wsl --status ---"
    wsl --status | Out-String | Select-Object -SkipLast 1
    Write-Host "--- wsl --list --verbose ---"
    wsl --list --verbose | Out-String | Select-Object -SkipLast 1
} catch {
    Write-Host "Error running WSL commands. Is WSL installed and enabled?"
}

# --- GPU Information (Best Effort) ---
Write-Host "`n[GPU INFORMATION (Best Effort via WMI)]"
try {
    $GpuInfo = Get-WmiObject Win32_VideoController -Filter "Name LIKE '%AMD%'"
    if ($GpuInfo) {
        foreach ($gpu in $GpuInfo) {
            Write-Host "GPU Name: $($gpu.Name)"
            Write-Host "Adapter RAM: $($([Math]::Round($gpu.AdapterRAM / 1GB, 2))) GB"
            Write-Host "Driver Version (WMI): $($gpu.DriverVersion)"
            Write-Host "Driver Date (WMI): $($gpu.DriverDate)"
            Write-Host "**NOTE:** WMI Driver Version may not directly match Adrenalin version (e.g., 25.8.1). Please verify manually in AMD Software."
        }
    } else {
        Write-Host "No AMD GPU detected via WMI."
    }
} catch {
    Write-Host "Error querying WMI for GPU information."
}

# --- Manual Check Reminders ---
Write-Host "`n=================================================="
Write-Host "MANUAL CHECKS REQUIRED:"
Write-Host "  1. Verify Windows AMD Driver Version:"
Write-Host "     - Open AMD Software: Adrenalin Edition application."
Write-Host "     - Go to System -> Software & Driver -> Driver Version."
Write-Host "     - CONFIRM it is exactly 'Adrenalin Edition 25.8.1 for WSL2' or the known required version."
Write-Host "  2. Verify Docker Desktop Status & WSL Integration:"
Write-Host "     - Ensure Docker Desktop application is running."
Write-Host "     - Go to Docker Desktop Settings -> Resources -> WSL Integration."
Write-Host "     - Ensure 'Enable integration with my default WSL distro' is ON."
Write-Host "     - Ensure the toggle for 'Ubuntu-22.04' (or your specific distro) is ON."
Write-Host "=================================================="

Write-Host "Report Complete."