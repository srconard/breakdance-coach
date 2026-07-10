<#
Runs the resilient Instagram movement-library ingest from Windows Task Scheduler.
The cookie jar lives outside the repository at
%LOCALAPPDATA%\breakdance-coach\ig_cookies.txt and is never committed or printed.

Scheduled task controls:
  schtasks /Change /TN "breakdance-ig-ingest" /DISABLE
  schtasks /Change /TN "breakdance-ig-ingest" /ENABLE
  schtasks /Delete /TN "breakdance-ig-ingest" /F
#>

$ErrorActionPreference = 'Continue'
$repo = "C:\dev\breakdance-coach"
Set-Location $repo

$stamp = Get-Date -Format 'yyyy-MM-dd_HHmmss'
$logDir = "output\ig_ingest\logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$logPath = "$logDir\nightly_$stamp.log"

cmd /c "python `"analyzer_3d\src\ig_ingest.py`" nightly --limit 45 > `"$logPath`" 2>&1"
exit $LASTEXITCODE
