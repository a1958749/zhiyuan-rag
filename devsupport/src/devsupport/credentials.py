"""Load the explicitly authorized Opus credential without logging secret material."""

import os
import subprocess
from pathlib import Path
from urllib.parse import urlsplit

TRUSTED_HOST = 'www.xiaoyaoapi.com'
ENV_NAME = 'DEVSUPPORT_OPUS_API_KEY'


def validate_host(base_url: str) -> str:
    """Return the normalized, allowlisted HTTPS v1 endpoint or reject it."""
    parsed = urlsplit(base_url)
    if (parsed.scheme != 'https' or parsed.hostname != TRUSTED_HOST or parsed.port not in (None, 443)
            or parsed.username or parsed.password or parsed.query or parsed.fragment
            or parsed.path.rstrip('/') != '/v1'):
        raise ValueError('provider endpoint is not allowlisted')
    return 'https://' + TRUSTED_HOST + '/v1'


def load_credential(path: str | Path = 'F:/DevSupportCopilot/.secrets/opus.dpapi',
                    env_override: bool = False) -> str:
    """Decrypt Windows-user DPAPI in a captured subprocess; opt in to one known env variable."""
    if env_override and os.environ.get(ENV_NAME):
        return os.environ[ENV_NAME]
    if os.name != 'nt':
        raise RuntimeError('DPAPI requires the owning Windows user; explicit env override is available')
    # Only the path enters the child environment. Neither ciphertext nor plaintext is a command argument.
    script = r'''
$ErrorActionPreference = 'Stop'
try {
    $raw = [IO.File]::ReadAllBytes($env:DEVSUPPORT_DPAPI_PATH)
    $text = [Text.Encoding]::UTF8.GetString($raw).Trim()
    if ($text -match '^01000000d08c9ddf') {
        $secure = ConvertTo-SecureString $text
        $ptr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
        try { [Console]::Write([Runtime.InteropServices.Marshal]::PtrToStringBSTR($ptr)) }
        finally { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($ptr) }
    } else {
        Add-Type -AssemblyName System.Security
        $plain = [Security.Cryptography.ProtectedData]::Unprotect($raw, $null,
            [Security.Cryptography.DataProtectionScope]::CurrentUser)
        [Console]::Write([Text.Encoding]::UTF8.GetString($plain))
        [Array]::Clear($plain, 0, $plain.Length)
    }
} catch { exit 1 }
'''
    env = dict(os.environ, DEVSUPPORT_DPAPI_PATH=str(Path(path).resolve()))
    failure = False
    try:
        result = subprocess.run(['powershell.exe', '-NoProfile', '-NonInteractive', '-Command', script],
                                capture_output=True, timeout=15, env=env, creationflags=0x08000000)
        secret = result.stdout.decode('utf-8').strip() if result.returncode == 0 else ''
        failure = not bool(secret)
    except Exception:
        failure, secret = True, ''
    if failure:
        raise RuntimeError('credential unavailable')
    return secret
