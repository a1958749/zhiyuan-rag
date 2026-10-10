param([switch]$ReuseBuild)
$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$frontend = Join-Path $repoRoot 'frontend'
$destination = Join-Path $repoRoot 'docsgpt/static'
if (-not $ReuseBuild) {
    Push-Location $frontend
    try {
        & npm.cmd run build
        if ($LASTEXITCODE -ne 0) { throw 'Frontend build failed.' }
    } finally { Pop-Location }
}
$dist = Join-Path $frontend 'dist'
if (-not (Test-Path -LiteralPath (Join-Path $dist 'index.html'))) { throw 'Build index is missing.' }
New-Item -ItemType Directory -Force -Path $destination | Out-Null
Get-ChildItem -LiteralPath $dist | Copy-Item -Destination $destination -Recurse -Force
$index = Join-Path $destination 'index.html'
$html = [IO.File]::ReadAllText($index)
if ($html -notmatch 'src="/config.js"') {
    $html = $html.Replace('<head>', '<head><script src="/config.js"></script>')
}
if ($html -notmatch 'src="/config.js"') { throw 'Runtime configuration injection failed.' }
[IO.File]::WriteAllText($index, $html, [Text.UTF8Encoding]::new($false))
Write-Output 'Built UI serves same-origin runtime configuration before its bundle.'
