$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$secretDirectory = Join-Path $repoRoot '.secrets'
New-Item -ItemType Directory -Force -Path $secretDirectory | Out-Null
$target = Join-Path $secretDirectory 'internal.dpapi'
if (Test-Path -LiteralPath $target) {
    Write-Output 'Existing internal credential retained.'
    exit 0
}
Add-Type -AssemblyName System.Security
$randomBytes = New-Object byte[] 32
$generator = [Security.Cryptography.RandomNumberGenerator]::Create()
try {
    $generator.GetBytes($randomBytes)
    $plain = [Text.Encoding]::UTF8.GetBytes([Convert]::ToBase64String($randomBytes))
    $encrypted = [Security.Cryptography.ProtectedData]::Protect($plain, $null,
        [Security.Cryptography.DataProtectionScope]::CurrentUser)
    $stream = [IO.File]::Open($target, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write)
    try { $stream.Write($encrypted, 0, $encrypted.Length) }
    finally { $stream.Dispose() }
} finally {
    $generator.Dispose()
    [Array]::Clear($randomBytes, 0, $randomBytes.Length)
    if ($plain) { [Array]::Clear($plain, 0, $plain.Length) }
}
Write-Output 'Prepared encrypted local API/worker credential.'
