<#
.SYNOPSIS
  Copies the "MV Bridge" CSP Lua app into the Assetto Corsa install.

.DESCRIPTION
  Run this yourself (Project MV tooling never writes into the game folder):

    powershell -ExecutionPolicy Bypass -File "D:\Visual Studio\Project MV\mv\ac\plugin\install_plugin.ps1"

  Optional: -AcRoot "X:\path\to\assettocorsa"   -Uninstall

  It copies  mv\ac\plugin\apps\lua\mv_bridge\*  to  <AcRoot>\apps\lua\mv_bridge\
  (overwriting older copies of the same two files) and nothing else.
#>
param(
    [string]$AcRoot = "E:\SteamLibrary\steamapps\common\assettocorsa",
    [switch]$Uninstall
)

$ErrorActionPreference = "Stop"
$src = Join-Path $PSScriptRoot "apps\lua\mv_bridge"
$dst = Join-Path $AcRoot "apps\lua\mv_bridge"

if (-not (Test-Path (Join-Path $AcRoot "acs.exe"))) {
    throw "acs.exe not found in '$AcRoot' - pass -AcRoot with your assettocorsa folder."
}
if (-not (Test-Path (Join-Path $AcRoot "extension"))) {
    Write-Warning "No 'extension' folder: Custom Shaders Patch does not seem to be installed. The app needs CSP."
}

if ($Uninstall) {
    if (Test-Path $dst) { Remove-Item -Recurse -Force $dst; Write-Host "Removed $dst" }
    else { Write-Host "Nothing to remove at $dst" }
    return
}

New-Item -ItemType Directory -Force -Path $dst | Out-Null
Copy-Item -Force (Join-Path $src "manifest.ini") $dst
Copy-Item -Force (Join-Path $src "mv_bridge.lua") $dst
Write-Host "Installed MV Bridge to $dst"
Get-ChildItem $dst | ForEach-Object { Write-Host ("  {0}  {1} bytes" -f $_.Name, $_.Length) }
