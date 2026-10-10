# Pure synthetic lifecycle checks; never launch or terminate a process.
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
Assert-Equal (Get-PipelineSuffix) ''
Assert-Equal (Get-PipelineSuffix -EnableFull) ' --enable-full'
Assert-Equal (Get-PipelineSuffix -EnableContextual) ' --enable-contextual'
Assert-Equal (Get-PipelineSuffix -EnableFull -EnableContextual) ' --enable-full --enable-contextual'
$entry = 'F:\fixture\serve_pipeline.py'
$executable = 'C:\Fixture\python.exe'
$global:ContextualFixtureProcess = [pscustomobject]@{ ProcessId = 321; ParentProcessId = 1; ExecutablePath = $executable;
    CommandLine = ''; CreationDate = [DateTime]::UtcNow }
function Get-CimInstance($ClassName, $Filter) {
    if ($Filter -eq 'ProcessId = 321') { return $global:ContextualFixtureProcess }
}
foreach ($suffix in @('', ' --enable-full', ' --enable-contextual', ' --enable-full --enable-contextual')) {
    $global:ContextualFixtureProcess.CommandLine = '"' + $executable + '" "' + $entry + '"' + $suffix
    $identities = @(Get-PipelineTree 321 $entry $suffix @($executable))
    Assert-Equal $identities.Count 1
    Assert-Equal $identities[0].command $global:ContextualFixtureProcess.CommandLine
}
foreach ($bad in @(' --unknown', ' --enable-contextual --enable-full', ' --enable-contextual --enable-contextual', ' --enable-full; echo x')) {
    $global:ContextualFixtureProcess.CommandLine = '"' + $executable + '" "' + $entry + '"' + $bad
    Assert-Rejected { Get-PipelineTree 321 $entry $bad @($executable) }
}
Write-Output ('Contextual lifecycle checks passed on PowerShell ' + $PSVersionTable.PSVersion.ToString())

# Run the actual launcher against copied scripts and fake process APIs only.
$scratch = Join-Path ([IO.Path]::GetTempPath()) ('devsupport-contextual-' + [Guid]::NewGuid().ToString('N'))
$fixtureScripts = Join-Path $scratch 'devsupport/scripts'
New-Item -ItemType Directory -Path $fixtureScripts -Force | Out-Null
Copy-Item -LiteralPath (Join-Path $repoRoot 'devsupport/scripts/start.ps1') -Destination $fixtureScripts
Copy-Item -LiteralPath (Join-Path $repoRoot 'devsupport/scripts/stop.ps1') -Destination $fixtureScripts
function Start-Process($FilePath, $ArgumentList, $WorkingDirectory, $WindowStyle,
                       [switch]$PassThru, $RedirectStandardOutput, $RedirectStandardError) {
    Assert-Equal $WindowStyle 'Hidden'
    Assert-Equal $WorkingDirectory $scratch
    $global:ContextualFixtureProcess.ExecutablePath = $FilePath
    $global:ContextualFixtureProcess.CommandLine = '"' + $FilePath + '" ' + ($ArgumentList -join ' ')
    return [pscustomobject]@{ Id = 321 }
}
function Get-Process($Id, $ErrorAction) {
    Assert-Equal $Id 321
    $fakeProcess = [pscustomobject]@{ Id = 321; HasExited = $true }
    $fakeProcess | Add-Member -MemberType ScriptMethod -Name WaitForExit -Value { param($milliseconds) return $true }
    return $fakeProcess
}
function Stop-Process($InputObject) {
    Assert-Equal $InputObject.Id 321
    $global:ContextualFixtureStops++
}
$fakePython = Join-Path $scratch 'fake/Scripts/python.exe'
foreach ($options in @(@{}, @{EnableFull=$true}, @{EnableContextual=$true}, @{EnableFull=$true;EnableContextual=$true})) {
    & (Join-Path $fixtureScripts 'start.ps1') -Python $fakePython @options | Out-Null
    $recordPath = Join-Path $scratch 'devsupport/runtime/pipeline-process.json'
    $record = Get-Content -LiteralPath $recordPath -Raw | ConvertFrom-Json
    Assert-Equal $record.suffix (Get-PipelineSuffix @options)
    Assert-Equal $record.processes[0].command $global:ContextualFixtureProcess.CommandLine
    $global:ContextualFixtureStops = 0
    & (Join-Path $fixtureScripts 'stop.ps1') | Out-Null
    Assert-Equal $global:ContextualFixtureStops 1
    Assert-Equal (Test-Path -LiteralPath $recordPath) $false
}
Write-Output 'Actual start/stop script flag/record checks passed with fake process APIs.'

