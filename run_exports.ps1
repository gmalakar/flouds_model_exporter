# Optional Windows convenience wrapper for batch exports.
# Canonical batch entrypoint: `flouds-export batch --preset recommended`.
# This script forwards to the Python CLI so batch orchestration has one implementation path.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File .\run_exports.ps1
#   powershell -ExecutionPolicy Bypass -File .\run_exports.ps1 -UseVenv -Force
#   powershell -ExecutionPolicy Bypass -File .\run_exports.ps1 -Config .\docs\batch_presets_example.yaml -Preset text-import
#   powershell -ExecutionPolicy Bypass -File .\run_exports.ps1 -TextFile .\docs\batch_commands.txt -Preset text-import
#   powershell -ExecutionPolicy Bypass -File .\run_exports.ps1 -BatchFile .\docs\batch_presets_full.yaml -Preset full
#
# Flags:
#   -UseVenv              : Require the repository .venv Python interpreter
#   -Force                : Apply --force to all batch exports
#   -Optimize             : Apply --optimize to all batch exports
#   -OptimizationLevel    : Apply --optimization-level to all batch exports; requires -Optimize
#   -Cleanup              : Apply --cleanup to all batch exports
#   -SkipValidator        : Apply --skip-validator to all batch exports unless active entries require validation
#   -PruneCanonical       : Apply --prune-canonical to all batch exports; requires -Cleanup
#   -Portable             : Apply --portable to all batch exports; requires -Optimize
#   -FailFast             : Stop batch on first failed export
#   -Config               : YAML/JSON batch config file
#   -TextFile             : Text file containing one `flouds-export export ...` command per line
#   -BatchFile            : YAML/JSON config or text command file, detected by extension
#   -Preset               : Preset name to run
#   -MinFreeMemoryGB      : Minimum free memory threshold
#   -LogToFile            : Request per-export log files
#
# Customize ONNX_PATH via env if needed: $env:ONNX_PATH = "onnx"

param(
    [switch]$UseVenv,
    [switch]$Force,
    [switch]$SkipValidator,
    [switch]$Optimize,
    [ValidateSet("0", "1", "2", "99")]
    [string]$OptimizationLevel,
    [switch]$Cleanup,
    [switch]$PruneCanonical,
    [switch]$Portable,
    [switch]$FailFast,
    [switch]$LogToFile,
    [string]$Config,
    [string]$TextFile,
    [string]$BatchFile,
    [string]$Preset = "recommended",
    [int]$MinFreeMemoryGB = 1
)

$ErrorActionPreference = 'Stop'
$presetWasProvided = $PSBoundParameters.ContainsKey('Preset')

if ($Portable -and -not $Optimize) {
    Write-Error "-Portable requires -Optimize."
    exit 1
}
if ($OptimizationLevel -and -not $Optimize) {
    Write-Error "-OptimizationLevel requires -Optimize."
    exit 1
}
if ($PruneCanonical -and -not $Cleanup) {
    Write-Error "-PruneCanonical requires -Cleanup."
    exit 1
}
if ($Config -and $TextFile) {
    Write-Error "Use only one of -Config or -TextFile."
    exit 1
}
if ($BatchFile -and ($Config -or $TextFile)) {
    Write-Error "Use -BatchFile by itself, or use -Config/-TextFile explicitly."
    exit 1
}

$repoRoot = $PSScriptRoot
$venvPythonExe = Join-Path $repoRoot ".venv\Scripts\python.exe"
if (Test-Path $venvPythonExe) {
    $pythonExe = $venvPythonExe
} elseif ($UseVenv) {
    Write-Error "Python venv not found at $venvPythonExe. Create/activate .venv first."
    exit 1
} else {
    $pythonExe = 'python'
}

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

function Get-FirstPresetName {
    param([string]$ConfigPath)

    $ext = [System.IO.Path]::GetExtension($ConfigPath).ToLowerInvariant()
    if ($ext -eq ".json") {
        try {
            $json = Get-Content -Raw -Path $ConfigPath -Encoding UTF8 | ConvertFrom-Json
            $props = $json.batch_presets.PSObject.Properties
            if ($props.Count -gt 0) {
                return $props[0].Name
            }
        } catch {
            return $null
        }
        return $null
    }

    $insidePresets = $false
    foreach ($raw in (Get-Content -Path $ConfigPath -Encoding UTF8)) {
        if ($raw -match '^\s*#') {
            continue
        }
        if ($raw -match '^\s*batch_presets\s*:\s*$') {
            $insidePresets = $true
            continue
        }
        if ($insidePresets -and $raw -match '^\s{2}([A-Za-z0-9_.-]+)\s*:\s*$') {
            return $Matches[1]
        }
    }
    return $null
}

function Split-CommandTokens {
    param([string]$Line)

    $tokens = @()
    $matches = [regex]::Matches($Line, '"([^"\\]|\\.)*"|''([^''\\]|\\.)*''|\S+')
    foreach ($m in $matches) {
        $token = $m.Value
        if (($token.StartsWith('"') -and $token.EndsWith('"')) -or ($token.StartsWith("'") -and $token.EndsWith("'"))) {
            $token = $token.Substring(1, $token.Length - 2)
        }
        $tokens += $token
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

    $tokens = $tokens[2..($tokens.Count - 1)]
    $entry = [ordered]@{}

    for ($i = 0; $i -lt $tokens.Count; $i++) {
        $token = $tokens[$i]
        switch ($token) {
            '--model-name' { if ($i + 1 -lt $tokens.Count) { $entry.model_name = $tokens[++$i] } else { throw "Missing value for $token" } }
            '--model-for' { if ($i + 1 -lt $tokens.Count) { $entry.model_for = $tokens[++$i] } else { throw "Missing value for $token" } }
            '--task' { if ($i + 1 -lt $tokens.Count) { $entry.task = $tokens[++$i] } else { throw "Missing value for $token" } }
            '--model-folder' { if ($i + 1 -lt $tokens.Count) { $entry.model_folder = $tokens[++$i] } else { throw "Missing value for $token" } }
            '--onnx-path' { if ($i + 1 -lt $tokens.Count) { $entry.onnx_path = $tokens[++$i] } else { throw "Missing value for $token" } }
            '--framework' { if ($i + 1 -lt $tokens.Count) { $entry.framework = $tokens[++$i] } else { throw "Missing value for $token" } }
            '--optimization-level' { if ($i + 1 -lt $tokens.Count) { $entry.optimization_level = [int]$tokens[++$i] } else { throw "Missing value for $token" } }
            '--opset-version' { if ($i + 1 -lt $tokens.Count) { $entry.opset_version = [int]$tokens[++$i] } else { throw "Missing value for $token" } }
            '--device' { if ($i + 1 -lt $tokens.Count) { $entry.device = $tokens[++$i] } else { throw "Missing value for $token" } }
            '--quantize' { if ($i + 1 -lt $tokens.Count) { $entry.quantize = $tokens[++$i] } else { throw "Missing value for $token" } }
            '--hf-token' { if ($i + 1 -lt $tokens.Count) { $entry.hf_token = $tokens[++$i] } else { throw "Missing value for $token" } }
            '--library' { if ($i + 1 -lt $tokens.Count) { $entry.library = $tokens[++$i] } else { throw "Missing value for $token" } }

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
            '--log-to-file' { $entry.log_to_file = $true }
            default {
                if ($token.StartsWith('--')) {
                    throw "Unsupported export flag in text file: $token"
                }
            }
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

    $config = [ordered]@{
        batch_presets = [ordered]@{
            $PresetName = $entries
        }
    }

    $tmpConfig = Join-Path ([System.IO.Path]::GetTempPath()) ("flouds_batch_{0}.json" -f [guid]::NewGuid().ToString('N'))
    $config | ConvertTo-Json -Depth 20 | Set-Content -Path $tmpConfig -Encoding UTF8
    return $tmpConfig
}

if ($BatchFile) {
    $batchPath = Resolve-RepoPath $BatchFile
    if (!(Test-Path $batchPath)) {
        Write-Error "Batch file not found at $batchPath"
        exit 1
    }

    $ext = [System.IO.Path]::GetExtension($batchPath).ToLowerInvariant()
    if ($ext -eq ".yaml" -or $ext -eq ".yml" -or $ext -eq ".json") {
        $Config = $batchPath
    } else {
        $TextFile = $batchPath
    }
}

$tempConfigPath = $null
$cliArgs = @("-m", "model_exporter.cli.main", "batch", "--min-free-memory-gb", $MinFreeMemoryGB.ToString())

if ($Config) {
    $configPath = Resolve-RepoPath $Config
    if (!(Test-Path $configPath)) {
        Write-Error "Batch config not found at $configPath"
        exit 1
    }
    if (-not $presetWasProvided) {
        $inferredPreset = Get-FirstPresetName $configPath
        if ($inferredPreset) {
            $Preset = $inferredPreset
        }
    }
    $cliArgs += @("--config", $configPath)
}

if ($TextFile) {
    $textPath = Resolve-RepoPath $TextFile
    if (!(Test-Path $textPath)) {
        Write-Error "Text file not found at $textPath"
        exit 1
    }
    if (-not $presetWasProvided -and $Preset -eq "recommended") {
        $Preset = "batch"
    }
    $tempConfigPath = Convert-TextFileToBatchConfig -InputTextPath $textPath -PresetName $Preset
    $cliArgs += @("--config", $tempConfigPath)
}

$cliArgs += @("--preset", $Preset)

if ($Force) { $cliArgs += "--force" }
if ($SkipValidator) { $cliArgs += "--skip-validator" }
if ($Optimize) { $cliArgs += "--optimize" }
if ($OptimizationLevel) { $cliArgs += @("--optimization-level", $OptimizationLevel) }
if ($Cleanup) { $cliArgs += "--cleanup" }
if ($PruneCanonical) { $cliArgs += "--prune-canonical" }
if ($Portable) { $cliArgs += "--portable" }
if ($FailFast) { $cliArgs += "--fail-fast" }
if ($LogToFile) { $cliArgs += "--log-to-file" }

$displayCmd = "$pythonExe $($cliArgs -join ' ')"
Write-Host "Running wrapper command: $displayCmd" -ForegroundColor Cyan

try {
    & $pythonExe @cliArgs
    $exitCode = $LASTEXITCODE
} catch {
    $exitCode = 1
    Write-Error "Batch execution failed: $($_.Exception.Message)"
} finally {
    if ($tempConfigPath -and (Test-Path $tempConfigPath)) {
        try {
            Remove-Item -Force $tempConfigPath
        } catch {
            Write-Warning "Could not remove temporary config file: $tempConfigPath"
        }
    }
}

exit $exitCode
