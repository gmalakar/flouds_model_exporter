# Optional Windows convenience wrapper for batch exports.
# Canonical batch entrypoint: `flouds-export batch --preset recommended`.
# This script forwards to the Python CLI so batch orchestration has one implementation path.
# Usage:
#   powershell -ExecutionPolicy Bypass -File .\run_exports.ps1
#   powershell -ExecutionPolicy Bypass -File .\run_exports.ps1 -UseVenv -Force
#   powershell -ExecutionPolicy Bypass -File .\run_exports.ps1 -Force -Optimize -Cleanup -SkipValidator -PruneCanonical -Portable -NoLocalPrep
#   powershell -ExecutionPolicy Bypass -File .\run_exports.ps1 -BatchFile .\docs\batch_presets_full.yaml
#
# Flags:
#   -UseVenv       : Use the repository .venv Python interpreter
#   -Force         : Append --force to each export command
#   -Optimize      : Append --optimize to each export command
#   -Cleanup       : Append --cleanup to each export command
#   -SkipValidator : Append --skip-validator to each export command
#   -PruneCanonical: Append --prune-canonical to each export command (optionally remove canonical ONNX files when merged artifacts exist)
#   -Portable      : Append --portable to each export command
#   -NoLocalPrep   : Append --no-local-prep to each export command
#   -FailFast      : Stop batch on first failed export
#   -BatchFile     : YAML/JSON batch config or text file of export commands
#
# Customize ONNX_PATH via env if needed: $env:ONNX_PATH = "onnx"

param(
    [switch]$UseVenv,
    [switch]$Force,
    [switch]$SkipValidator,
    [switch]$Optimize,
    [switch]$Cleanup,
    [switch]$PruneCanonical,
    [switch]$NoLocalPrep,
    [switch]$Portable,
    [switch]$FailFast,
    [switch]$LogToFile,
    [string]$BatchFile,
    [int]$MinFreeMemoryGB = 1
)

$ErrorActionPreference = 'Stop'

# Determine Python command (optionally use workspace .venv)
$repoRoot = $PSScriptRoot
if ($UseVenv) {
    $pythonExe = Join-Path $repoRoot ".venv\Scripts\python.exe"
    if (!(Test-Path $pythonExe)) {
        Write-Error "Python venv not found at $pythonExe. Create/activate .venv first."; exit 1
    }
} else {
    $pythonExe = 'python'
}

# Ensure src-layout package imports work from repo checkout even when
# editable install has not been performed yet.
$srcPath = Join-Path $repoRoot "src"
if (Test-Path $srcPath) {
    if ([string]::IsNullOrWhiteSpace($env:PYTHONPATH)) {
        $env:PYTHONPATH = $srcPath
    } else {
        $env:PYTHONPATH = "$srcPath;$($env:PYTHONPATH)"
    }
}

function Resolve-RepoPath {
    param([string]$InputPath)
    if (-not $InputPath) {
        return $null
    }
    if ([System.IO.Path]::IsPathRooted($InputPath)) {
        return $InputPath
    }
    return (Join-Path $repoRoot $InputPath)
}

function Split-CommandTokens {
    param([string]$Line)

    $tokens = @()
    $matches = [regex]::Matches($Line, '"([^"\\]|\\.)*"|''([^''\\]|\\.)*''|\S+')
    foreach ($m in $matches) {
        $t = $m.Value
        if (($t.StartsWith('"') -and $t.EndsWith('"')) -or ($t.StartsWith("'") -and $t.EndsWith("'"))) {
            $t = $t.Substring(1, $t.Length - 2)
        }
        $tokens += $t
    }
    return $tokens
}

function Parse-ExportLine {
    param([string]$RawLine)

    $line = $RawLine.Trim()
    if (-not $line -or $line.StartsWith('#')) {
        return $null
    }

    $line = $line.TrimEnd(',')
    if (($line.StartsWith('"') -and $line.EndsWith('"')) -or ($line.StartsWith("'") -and $line.EndsWith("'"))) {
        $line = $line.Substring(1, $line.Length - 2)
    }

    if ($line -notmatch '^flouds-export\s+export(?:\s+|$)') {
        return $null
    }

    $tokens = Split-CommandTokens $line
    if ($tokens.Count -lt 2) {
        return $null
    }

    # Drop the required command prefix and keep only export flags.
    $tokens = $tokens[2..($tokens.Count - 1)]

    $entry = [ordered]@{}

    for ($i = 0; $i -lt $tokens.Count; $i++) {
        $tok = $tokens[$i]
        switch ($tok) {
            '--model-name' { if ($i + 1 -lt $tokens.Count) { $entry.model_name = $tokens[++$i] } }
            '--model-for' { if ($i + 1 -lt $tokens.Count) { $entry.model_for = $tokens[++$i] } }
            '--task' { if ($i + 1 -lt $tokens.Count) { $entry.task = $tokens[++$i] } }
            '--model-folder' { if ($i + 1 -lt $tokens.Count) { $entry.model_folder = $tokens[++$i] } }
            '--onnx-path' { if ($i + 1 -lt $tokens.Count) { $entry.onnx_path = $tokens[++$i] } }
            '--framework' { if ($i + 1 -lt $tokens.Count) { $entry.framework = $tokens[++$i] } }
            '--opset-version' { if ($i + 1 -lt $tokens.Count) { $entry.opset_version = [int]$tokens[++$i] } }
            '--device' { if ($i + 1 -lt $tokens.Count) { $entry.device = $tokens[++$i] } }
            '--quantize' { if ($i + 1 -lt $tokens.Count) { $entry.quantize = $tokens[++$i] } }
            '--pack-single-threshold-mb' { if ($i + 1 -lt $tokens.Count) { $entry.pack_single_threshold_mb = [int]$tokens[++$i] } }
            '--hf-token' { if ($i + 1 -lt $tokens.Count) { $entry.hf_token = $tokens[++$i] } }
            '--library' { if ($i + 1 -lt $tokens.Count) { $entry.library = $tokens[++$i] } }

            '--optimize' { $entry.optimize = $true }
            '--trust-remote-code' { $entry.trust_remote_code = $true }
            '--normalize-embeddings' { $entry.normalize_embeddings = $true }
            '--require-validator' { $entry.require_validator = $true }
            '--skip-validator' { $entry.skip_validator = $true }
            '--force' { $entry.force = $true }
            '--pack-single-file' { $entry.pack_single_file = $true }
            '--use-external-data-format' { $entry.use_external_data_format = $true }
            '--no-local-prep' { $entry.no_local_prep = $true }
            '--merge' { $entry.merge = $true }
            '--cleanup' { $entry.cleanup = $true }
            '--prune-canonical' { $entry.prune_canonical = $true }
            '--no-post-process' { $entry.no_post_process = $true }
            '--portable' { $entry.portable = $true }
            '--use-sub-process' { $entry.use_subprocess = $true }
            '--low-memory-env' { $entry.low_memory_env = $true }
        }
    }

    if (-not ($entry.Keys -contains 'model_name')) {
        return $null
    }
    if (-not ($entry.Keys -contains 'model_for')) {
        $entry.model_for = 'fe'
    }
    if (-not ($entry.Keys -contains 'task')) {
        $entry.task = 'feature-extraction'
    }

    return $entry
}

function Convert-TextFileToBatchConfig {
    param(
        [string]$InputTextPath,
        [string]$PresetName
    )

    $entries = New-Object System.Collections.Generic.List[object]
    foreach ($raw in (Get-Content -Path $InputTextPath -Encoding UTF8)) {
        $entry = Parse-ExportLine $raw
        if ($null -ne $entry) {
            $entries.Add([pscustomobject]$entry)
        }
    }

    if ($entries.Count -eq 0) {
        Write-Error "No valid export lines were found in text file: $InputTextPath"
        exit 1
    }

    $cfg = [ordered]@{
        batch_presets = [ordered]@{
            $PresetName = $entries
        }
    }

    $tmpConfig = Join-Path ([System.IO.Path]::GetTempPath()) ("flouds_batch_{0}.json" -f [guid]::NewGuid().ToString('N'))
    $cfg | ConvertTo-Json -Depth 20 | Set-Content -Path $tmpConfig -Encoding UTF8
    return $tmpConfig
}


$cliArgs = @("-m", "model_exporter.cli.main", "batch", "--min-free-memory-gb", $MinFreeMemoryGB.ToString())


$tempConfigPath = $null
if ($BatchFile) {
    $batchPath = Resolve-RepoPath $BatchFile
    if (!(Test-Path $batchPath)) {
        Write-Error "Batch file not found at $batchPath"; exit 1
    }
    $ext = [System.IO.Path]::GetExtension($batchPath).ToLowerInvariant()
    if ($ext -eq ".yaml" -or $ext -eq ".yml" -or $ext -eq ".json") {
        $cliArgs += @("--config", $batchPath)
    } else {
        $presetName = "batch"
        $tempConfigPath = Convert-TextFileToBatchConfig -InputTextPath $batchPath -PresetName $presetName
        $cliArgs += @("--config", $tempConfigPath, "--preset", $presetName)
    }
}



if ($Force) { $cliArgs += "--force" }
if ($SkipValidator) { $cliArgs += "--skip-validator" }
if ($Optimize) { $cliArgs += "--optimize" }
if ($Cleanup) { $cliArgs += "--cleanup" }
if ($PruneCanonical) { $cliArgs += "--prune-canonical" }
if ($NoLocalPrep) { $cliArgs += "--no-local-prep" }
if ($Portable) { $cliArgs += "--portable" }
if ($FailFast) { $cliArgs += "--fail-fast" }
if ($LogToFile) { $cliArgs += "--log-to-file" }

$displayCmd = "$pythonExe $($cliArgs -join ' ')"
Write-Host "Running wrapper command: $displayCmd" -ForegroundColor Cyan

try {
    & $pythonExe @cliArgs
    $exitCode = $LASTEXITCODE
}
catch {
    $exitCode = 1
    Write-Error "Batch execution failed: $($_.Exception.Message)"
}
finally {
    if ($tempConfigPath -and (Test-Path $tempConfigPath)) {
        try {
            Remove-Item -Force $tempConfigPath
        }
        catch {
            Write-Warning "Could not remove temporary config file: $tempConfigPath"
        }
    }
}

exit $exitCode
