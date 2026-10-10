param(
    [string]$Python = 'F:/DevSupportCopilotWork/.venv-core/Scripts/python.exe',
    [switch]$EnableFull,
    [switch]$EnableContextual
)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'stop.ps1') -FunctionsOnly
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$runtime = Join-Path $repoRoot 'devsupport/runtime'
New-Item -ItemType Directory -Force -Path $runtime | Out-Null
$recordPath = Join-Path $runtime 'pipeline-process.json'
if (Test-Path -LiteralPath $recordPath) { throw 'Existing process record; verify/stop it first.' }
$entry = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot 'serve_pipeline.py'))
$arguments = @(('"' + $entry + '"'))
if ($EnableFull) { $arguments += '--enable-full' }
if ($EnableContextual) { $arguments += '--enable-contextual' }
$child = Start-Process -FilePath $Python -ArgumentList $arguments -WorkingDirectory $repoRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $runtime 'pipeline.stdout.log') -RedirectStandardError (Join-Path $runtime 'pipeline.stderr.log')
$identity = Get-CimInstance Win32_Process -Filter "ProcessId = $($child.Id)"
$suffix = Get-PipelineSuffix -EnableFull:$EnableFull -EnableContextual:$EnableContextual
$allowed = @(Get-PipelineExecutables $identity.ExecutablePath)
@{ version = 2; suffix = $suffix; allowedExecutables = $allowed; processes = @((Get-PipelineIdentity $identity)) } | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $recordPath
# Capture the Windows venv launcher and its actual base-interpreter child.
$processes = @()
for ($attempt = 0; $attempt -lt 50; $attempt++) {
    $processes = @(Get-PipelineTree $child.Id $entry $suffix $allowed)
    if ($processes.Count -gt 1 -or $allowed.Count -eq 1) { break }
    Start-Sleep -Milliseconds 100
}
if ($processes.Count -eq 0) { throw 'Pipeline exited before its identity could be recorded.' }
@{ version = 2; suffix = $suffix; allowedExecutables = $allowed; processes = $processes } | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $recordPath
Write-Output "Pipeline process $($child.Id); health: http://127.0.0.1:8094/health"
