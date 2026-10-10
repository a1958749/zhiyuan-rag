$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
. (Join-Path $repoRoot 'devsupport/scripts/stop.ps1') -FunctionsOnly

function Assert-Equal($Actual, $Expected) {
    if ($Actual -cne $Expected) { throw "Assertion failed: $Actual != $Expected" }
}
function Assert-Rejected([scriptblock]$Action) {
    $rejected = $false
    try { & $Action } catch { $rejected = $true }
    if (-not $rejected) { throw 'Expected fail-closed rejection.' }
}

$iso = '2026-10-08T21:13:10.4329890+08:00'
$date = [DateTimeOffset]::Parse($iso).LocalDateTime
Assert-Equal (Get-PipelineStamp $iso) (Get-PipelineStamp $date)
$json = @{ created = $iso } | ConvertTo-Json | ConvertFrom-Json
Assert-Equal (Get-PipelineStamp $json.created) (Get-PipelineStamp $date)
$stamp = Get-PipelineStamp $date
$json = @{ created = $stamp } | ConvertTo-Json | ConvertFrom-Json
Assert-Equal $json.created $stamp
Assert-Equal $json.created.GetType().FullName 'System.String'

$live = [pscustomobject]@{ ProcessId = 123; ParentProcessId = 100; ExecutablePath = 'C:\Python\python.exe';
    CommandLine = '"C:\Python\python.exe" "F:\repo\serve_pipeline.py"'; CreationDate = $date }
$saved = Get-PipelineIdentity $live
Assert-PipelineIdentity $live $saved
foreach ($field in @('pid', 'parent', 'command', 'executable', 'created')) {
    $changed = $saved | ConvertTo-Json | ConvertFrom-Json
    switch ($field) {
        'pid' { $changed.pid = 124 }
        'parent' { $changed.parent = 101 }
        'created' { $changed.created = 'ticks:1' }
        default { $changed.$field = 'different' }
    }
    Assert-Rejected { Assert-PipelineIdentity $live $changed }
}

# Read-only validation of the actual running v2 launcher/worker record.
$record = Get-Content -Raw (Join-Path $repoRoot 'devsupport/runtime/pipeline-process.json') | ConvertFrom-Json
Assert-Equal ([string]$record.version) '2'
if (@($record.processes).Count -lt 2) { throw 'Venv launcher child was not recorded.' }
foreach ($item in $record.processes) {
    Assert-Equal $item.created.GetType().FullName 'System.String'
    $running = Get-CimInstance Win32_Process -Filter "ProcessId = $($item.pid)"
    if (-not $running) { throw 'Recorded pipeline process is not live.' }
    Assert-PipelineIdentity $running $item
}
$entry = (Resolve-Path (Join-Path $repoRoot 'devsupport/scripts/serve_pipeline.py')).Path
Assert-Rejected {
    Get-PipelineTree $record.processes[0].pid ($entry + '.unknown') $record.suffix $record.allowedExecutables
}
Assert-Rejected {
    Get-PipelineTree $record.processes[0].pid $entry $record.suffix @('C:\unapproved\python.exe')
}
Write-Output ('Lifecycle identity checks passed on PowerShell ' + $PSVersionTable.PSVersion.ToString())
