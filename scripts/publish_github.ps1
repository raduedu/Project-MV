<#
Publish Project MV to GitHub as a PRIVATE repository named "Project-MV"
(GitHub does not allow spaces in repository names), or push new commits to it.

Run from the project folder in PowerShell:

    powershell -ExecutionPolicy Bypass -File scripts\publish_github.ps1

What it does, step by step (it stops and explains if anything is missing):
  1. Checks that the GitHub CLI ("gh") is installed. If not, it tells you the
     one command to install it (it does NOT install anything by itself).
  2. Checks you are logged in to GitHub ("gh auth login" opens your browser once).
  3. Safety check: refuses to publish if any Assetto Corsa data, training runs
     or model checkpoints are tracked by git (they are git-ignored on purpose).
  4. First time: creates the private repository from this folder and pushes.
     Later times: pushes new commits.
#>
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$repo = "Project-MV"

if (-not (Get-Command gh -ErrorAction SilentlyContinue)) {
    Write-Host "GitHub CLI not found. Install it once with:" -ForegroundColor Yellow
    Write-Host "    winget install --id GitHub.cli -e"
    Write-Host "then open a NEW PowerShell window and run this script again."
    exit 1
}

$loggedIn = $false
try { gh auth status 2>$null | Out-Null; $loggedIn = ($LASTEXITCODE -eq 0) } catch { $loggedIn = $false }
if (-not $loggedIn) {
    Write-Host "Not logged in to GitHub. Starting the login (choose GitHub.com, HTTPS, browser)..." -ForegroundColor Yellow
    gh auth login
    if ($LASTEXITCODE -ne 0) { Write-Host "Login did not complete."; exit 1 }
}

# --- safety: nothing private or bulky may be tracked -------------------------------
$bad = git ls-files | Select-String -Pattern '(^|/)(vehicles/local|tracks/local|\.cache|runs)/|\.pt$|\.npz$|data\.acd$|\.kn5$|(^|/)CLAUDE\.md$'
if ($bad) {
    Write-Host "Refusing to publish: these tracked files must stay local:" -ForegroundColor Red
    $bad | ForEach-Object { Write-Host "   $_" }
    exit 1
}
if (git status --porcelain) {
    Write-Host "Note: there are uncommitted changes; only committed work is published." -ForegroundColor Yellow
}

$hasRemote = (git remote) -contains "origin"
if (-not $hasRemote) {
    $user = gh api user --jq .login
    # "not found" is the normal answer the first time; with ErrorActionPreference=Stop,
    # Windows PowerShell 5.1 would turn gh's error message into a fatal error, hence try/catch
    $exists = $false
    try { gh repo view "$user/$repo" 2>$null | Out-Null; $exists = ($LASTEXITCODE -eq 0) } catch { $exists = $false }
    if ($exists) {
        Write-Host "Repository $user/$repo already exists; connecting this folder to it."
        git remote add origin "https://github.com/$user/$repo.git"
        git push -u origin HEAD
    } else {
        gh repo create $repo --private --source . --remote origin --push `
            --description "Project MV - modular, car-agnostic autonomous racing driver (reinforcement learning)"
    }
} else {
    git push origin HEAD
}
if ($LASTEXITCODE -eq 0) {
    $url = gh repo view --json url --jq .url
    $vis = gh repo view --json visibility --jq .visibility
    Write-Host "Done: $url  (visibility: $vis)" -ForegroundColor Green
}
