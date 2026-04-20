param(
    [switch]$PrintOnly,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs
)

$ErrorActionPreference = "Stop"

$exampleDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$localRepoRoot = (Resolve-Path (Join-Path $exampleDir "..\\..\\..\\..")).Path

$physicsnemoCandidates = @(
    $localRepoRoot,
    "C:\\Users\\jrj6wm\\OneDrive - University of Virginia\\Cursor\\modulus"
)

$runtimeRepoRoot = $null
foreach ($candidate in $physicsnemoCandidates) {
    if (-not (Test-Path -LiteralPath $candidate)) {
        continue
    }

    $hasCheckpointUtils = Test-Path -LiteralPath (Join-Path $candidate "physicsnemo\\launch\\utils\\checkpoint.py")
    $hasFilesystemUtils = Test-Path -LiteralPath (Join-Path $candidate "physicsnemo\\utils\\filesystem.py")
    if ($hasCheckpointUtils -and $hasFilesystemUtils) {
        $runtimeRepoRoot = (Resolve-Path $candidate).Path
        break
    }
}

if (-not $runtimeRepoRoot) {
    throw "Could not find a PhysicsNeMo runtime root with launch/checkpoint and utils/filesystem support."
}

$sourceRoot = (Join-Path $exampleDir "smoke_data\\source")
$targetTrainRoot = (Join-Path $exampleDir "smoke_data\\target")
$targetRolloutRoot = "C:/Users/jrj6wm/Box/Flood_Modeling/Simulations/Case_4/Results_Target/Test_20_Paper"

$requiredPaths = @(
    $sourceRoot,
    (Join-Path $sourceRoot "train.txt"),
    $targetTrainRoot,
    (Join-Path $targetTrainRoot "train.txt"),
    $targetRolloutRoot
)

foreach ($path in $requiredPaths) {
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Required path is missing: $path"
    }
}

Get-Command python | Out-Null

if ([string]::IsNullOrWhiteSpace($env:PYTHONPATH)) {
    $env:PYTHONPATH = $runtimeRepoRoot
} else {
    $pythonPathEntries = $env:PYTHONPATH -split ";"
    if ($pythonPathEntries -notcontains $runtimeRepoRoot) {
        $env:PYTHONPATH = "$runtimeRepoRoot;$env:PYTHONPATH"
    }
}

$command = @("python", "train.py", "--config-name", "config_smoke") + $ExtraArgs

Write-Host "Example directory: $exampleDir"
Write-Host "Local repo root: $localRepoRoot"
Write-Host "PhysicsNeMo runtime root: $runtimeRepoRoot"
Write-Host "Source train root: $sourceRoot"
Write-Host "Target train root: $targetTrainRoot"
Write-Host "Target rollout root: $targetRolloutRoot"
Write-Host "Command: $($command -join ' ')"

if ($PrintOnly) {
    exit 0
}

Push-Location $exampleDir
try {
    & $command[0] $command[1..($command.Count - 1)]
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
