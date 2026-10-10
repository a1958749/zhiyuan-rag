param([switch]$FunctionsOnly)
$ErrorActionPreference = 'Stop'

function Get-PipelineSuffix([switch]$EnableFull, [switch]$EnableContextual) {
    $suffix = ''
    if ($EnableFull) { $suffix += ' --enable-full' }
    if ($EnableContextual) { $suffix += ' --enable-contextual' }
    return $suffix
}

function Assert-PipelineSuffix([string]$Suffix) {
    if ($Suffix -cnotin @('', ' --enable-full', ' --enable-contextual', ' --enable-full --enable-contextual')) {
        throw 'Unknown pipeline command options; refusing lifecycle action.'
    }
}

function Get-PipelineStamp($Value) {
    if ($Value -is [string] -and $Value.StartsWith('ticks:')) { return $Value }
    if ($Value -is [DateTime]) { return 'ticks:' + $Value.ToUniversalTime().Ticks.ToString() }
    return 'ticks:' + ([DateTimeOffset]::Parse([string]$Value, [Globalization.CultureInfo]::InvariantCulture)).UtcDateTime.Ticks.ToString()
}

function Get-PipelineIdentity($Process) {
    return [pscustomobject]@{
        pid = [int]$Process.ProcessId
        parent = [int]$Process.ParentProcessId
        executable = [string]$Process.ExecutablePath
        command = [string]$Process.CommandLine
        created = Get-PipelineStamp $Process.CreationDate
    }
}

function Assert-PipelineIdentity($Live, $Saved) {
    if ($Live.CommandLine -cne $Saved.command -or $Live.ExecutablePath -ine $Saved.executable -or
        (Get-PipelineStamp $Live.CreationDate) -ne (Get-PipelineStamp $Saved.created) -or
        [int]$Live.ProcessId -ne [int]$Saved.pid) {
        throw 'Process identity changed; refusing to stop.'
    }
    if ($Saved.PSObject.Properties['parent'] -and [int]$Live.ParentProcessId -ne [int]$Saved.parent) {
        throw 'Process parent identity changed; refusing to stop.'
    }
}

function Get-PipelineExecutables([string]$Executable) {
    $allowed = @($Executable)
    $config = Join-Path (Split-Path (Split-Path $Executable -Parent) -Parent) 'pyvenv.cfg'
    if (Test-Path -LiteralPath $config) {
        $match = @(Get-Content -LiteralPath $config | Where-Object { $_ -match '^home\s*=\s*(.+)$' })
        if ($match.Count -eq 1) {
            $baseHome = ($match[0] -replace '^home\s*=\s*', '').Trim()
            $allowed += [IO.Path]::GetFullPath((Join-Path $baseHome 'python.exe'))
        }
    }
    return $allowed
}

function Get-PipelineTree([int]$RootId, [string]$Entry, [string]$Suffix, [string[]]$Allowed) {
    Assert-PipelineSuffix $Suffix
    $queue = New-Object 'System.Collections.Generic.Queue[int]'
    $queue.Enqueue($RootId)
    while ($queue.Count -gt 0) {
        $currentId = $queue.Dequeue()
        $live = Get-CimInstance Win32_Process -Filter "ProcessId = $currentId"
        if (-not $live) { continue }
        # Start-Process's hidden console is owned by Windows, not a worker.
        # Recognize only the exact system helper and let Windows reap it when
        # its Python clients exit; never terminate a console host explicitly.
        $consolePath = Join-Path $env:SystemRoot 'system32/conhost.exe'
        if ($live.ExecutablePath -ieq $consolePath -and
            $live.CommandLine -ieq ('\??\' + $consolePath + ' 0x4')) {
            if (@(Get-CimInstance Win32_Process -Filter "ParentProcessId = $currentId").Count -ne 0) {
                throw 'Unexpected console-host descendants; refusing lifecycle action.'
            }
            continue
        }
        $expected = '"' + $live.ExecutablePath + '" "' + $Entry + '"' + $Suffix
        # Windows PowerShell 5.1 Start-Process appends one trailing space.
        if ($live.ExecutablePath -notin $Allowed -or
            ($live.CommandLine -cne $expected -and $live.CommandLine -cne ($expected + ' '))) {
            throw "Unknown pipeline descendant command or executable (PID $currentId); refusing lifecycle action."
        }
        Get-PipelineIdentity $live
        foreach ($child in @(Get-CimInstance Win32_Process -Filter "ParentProcessId = $currentId")) {
            if ($child.CreationDate -lt $live.CreationDate) { throw 'Invalid pipeline descendant creation time.' }
            $queue.Enqueue([int]$child.ProcessId)
        }
    }
}

if ($FunctionsOnly) { return }
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$recordPath = Join-Path $repoRoot 'devsupport/runtime/pipeline-process.json'
$record = Get-Content -Raw -LiteralPath $recordPath | ConvertFrom-Json
$entry = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot 'serve_pipeline.py'))

if ($record.PSObject.Properties['processes']) {
    $saved = @($record.processes)
    $suffix = [string]$record.suffix
    $allowed = @($record.allowedExecutables)
} else {
    $rootLive = Get-CimInstance Win32_Process -Filter "ProcessId = $($record.pid)"
    if (-not $rootLive) { throw 'Legacy root is absent; descendants cannot be verified.' }
    Assert-PipelineIdentity $rootLive $record
    $prefix = '"' + $record.executable + '" "' + $entry + '"'
    if (-not $record.command.StartsWith($prefix, [StringComparison]::Ordinal)) { throw 'Unknown legacy command.' }
    $suffix = $record.command.Substring($prefix.Length)
    Assert-PipelineSuffix $suffix
    $allowed = @(Get-PipelineExecutables $record.executable)
    $saved = @(Get-PipelineTree $record.pid $entry $suffix $allowed)
}

Assert-PipelineSuffix $suffix
if ($saved.Count -eq 0) { throw 'Empty process record; refusing lifecycle action.' }
foreach ($item in $saved) {
    $live = Get-CimInstance Win32_Process -Filter "ProcessId = $($item.pid)"
    if (-not $live) { continue }
    Assert-PipelineIdentity $live $item
    foreach ($observed in @(Get-PipelineTree $item.pid $entry $suffix $allowed)) {
        $match = @($saved | Where-Object { $_.pid -eq $observed.pid })
        if ($match.Count -ne 1) { throw 'Unrecorded descendant; refusing to stop.' }
        $childLive = Get-CimInstance Win32_Process -Filter "ProcessId = $($observed.pid)"
        if ($childLive) { Assert-PipelineIdentity $childLive $match[0] }
    }
}
for ($index = $saved.Count - 1; $index -ge 0; $index--) {
    $item = $saved[$index]
    $live = Get-CimInstance Win32_Process -Filter "ProcessId = $($item.pid)"
    if (-not $live) { continue }
    Assert-PipelineIdentity $live $item
    $target = Get-Process -Id $item.pid -ErrorAction Stop
    Stop-Process -InputObject $target
    $target.WaitForExit(10000) | Out-Null
    if (-not $target.HasExited) { throw 'Pipeline process did not exit; retaining identity record.' }
}
Remove-Item -LiteralPath $recordPath
Write-Output 'Stopped the verified pipeline process tree.'
