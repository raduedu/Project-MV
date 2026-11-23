<#
.SYNOPSIS
  Copies the "MV Bridge" and "MV Viewer" CSP Lua apps into the Assetto Corsa install.

.DESCRIPTION
  Run this yourself (Project MV tooling never writes into the game folder):

    powershell -ExecutionPolicy Bypass -File "D:\Visual Studio\Project MV\mv\ac\plugin\install_plugin.ps1"

  Optional: -AcRoot "X:\path\to\assettocorsa"   -Uninstall

  It copies  mv\ac\plugin\apps\lua\mv_bridge\*  to  <AcRoot>\apps\lua\mv_bridge\
  and        mv\ac\plugin\apps\lua\mv_viewer\*  to  <AcRoot>\apps\lua\mv_viewer\
  (overwriting older copies of the same files) and nothing else.
#>
param(
    [string]$AcRoot = "E:\SteamLibrary\steamapps\common\assettocorsa",
    [switch]$Uninstall
)

$ErrorActionPreference = "Stop"
$apps = @("mv_bridge", "mv_viewer")

if (-not (Test-Path (Join-Path $AcRoot "acs.exe"))) {
    throw "acs.exe not found in '$AcRoot' - pass -AcRoot with your assettocorsa folder."
}
if (-not (Test-Path (Join-Path $AcRoot "extension"))) {
    Write-Warning "No 'extension' folder: Custom Shaders Patch does not seem to be installed. The apps need CSP."
}

foreach ($app in $apps) {
    $src = Join-Path $PSScriptRoot "apps\lua\$app"
    $dst = Join-Path $AcRoot "apps\lua\$app"
    if ($Uninstall) {
        if (Test-Path $dst) { Remove-Item -Recurse -Force $dst; Write-Host "Removed $dst" }
        else { Write-Host "Nothing to remove at $dst" }
        continue
    }
    New-Item -ItemType Directory -Force -Path $dst | Out-Null
    Copy-Item -Force (Join-Path $src "manifest.ini") $dst
    Copy-Item -Force (Join-Path $src "$app.lua") $dst
    Write-Host "Installed $app to $dst"
    Get-ChildItem $dst | ForEach-Object { Write-Host ("  {0}  {1} bytes" -f $_.Name, $_.Length) }
}
