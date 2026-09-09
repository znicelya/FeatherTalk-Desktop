#Requires -Version 5.1
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$MsiPath,
    [switch]$SkipGuiSmoke
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
. (Join-Path $PSScriptRoot 'models.ps1')
$MsiPath = (Resolve-Path -LiteralPath $MsiPath).Path
$rustRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$manifestPath = [IO.Path]::ChangeExtension($MsiPath, 'payload.json')
$manifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
if (!$manifest.PSObject.Properties['models'] -or $manifest.models.Count -ne 4) {
    throw 'The MSI must declare all four bundled model packages, including VGG19.'
}
$expectedMsiHash = ((Get-Content -LiteralPath "$MsiPath.sha256" -Raw).Trim() -split '\s+')[0]
if ((Get-FileHash -LiteralPath $MsiPath -Algorithm SHA256).Hash -ne $expectedMsiHash) {
    throw 'The MSI SHA-256 checksum does not match its sidecar.'
}

function Get-MsiRows {
    param($Database, [string]$Sql, [string[]]$Columns)
    $view = $Database.GetType().InvokeMember('OpenView', 'InvokeMethod', $null, $Database, @($Sql))
    try {
        $view.GetType().InvokeMember('Execute', 'InvokeMethod', $null, $view, @()) | Out-Null
        while ($record = $view.GetType().InvokeMember('Fetch', 'InvokeMethod', $null, $view, @())) {
            try {
                $row = [ordered]@{}
                for ($column = 0; $column -lt $Columns.Count; $column++) {
                    $row[$Columns[$column]] = $record.GetType().InvokeMember('StringData', 'GetProperty', $null, $record, @($column + 1))
                }
                [pscustomobject]$row
            } finally { [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($record) }
        }
    } finally {
        $view.GetType().InvokeMember('Close', 'InvokeMethod', $null, $view, @()) | Out-Null
        [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($view)
    }
}

function ConvertTo-NativeArgument {
    param([string]$Value)
    '"' + [regex]::Replace([regex]::Replace($Value, '(\\*)"', '$1$1\"'), '(\\+)$', '$1$1') + '"'
}

function Get-RelativeMsiDirectory {
    param([string]$Id, [hashtable]$Directories)
    $parts = @()
    $seen = @{}
    while ($Id -ne 'INSTALLFOLDER') {
        if (!$Directories.ContainsKey($Id) -or $seen.ContainsKey($Id)) {
            throw "MSI payload directory does not descend from INSTALLFOLDER: $Id"
        }
        $seen[$Id] = $true
        $directory = $Directories[$Id]
        $name = (($directory.Name -split ':')[0] -split '\|')[-1]
        if ($name -ne '.') { $parts = @($name) + $parts }
        $Id = $directory.Parent
    }
    return ($parts -join '/')
}

function New-CleanProcess {
    param([string]$Executable, [string[]]$ToolArguments, [string]$Directory)
    $info = [Diagnostics.ProcessStartInfo]::new()
    $info.FileName = $Executable
    $info.Arguments = ($ToolArguments | ForEach-Object { ConvertTo-NativeArgument $_ }) -join ' '
    $info.WorkingDirectory = $Directory
    $info.UseShellExecute = $false
    $info.CreateNoWindow = $true
    $info.RedirectStandardInput = $true
    $info.RedirectStandardOutput = $true
    $info.RedirectStandardError = $true
    foreach ($key in @($info.EnvironmentVariables.Keys)) {
        if ($key -like 'FEATHERTALK_*') { $info.EnvironmentVariables.Remove($key) }
    }
    $info.EnvironmentVariables['PATH'] = "$env:SystemRoot\System32;$env:SystemRoot"
    $info.EnvironmentVariables['FEATHERTALK_WORKER_BACKEND'] = 'cpu'
    $process = [Diagnostics.Process]::new()
    $process.StartInfo = $info
    if (!$process.Start()) { throw "Could not start $Executable" }
    $process.StandardInput.Close()
    return $process
}

function Invoke-CleanProcess {
    param([string]$Executable, [string[]]$ToolArguments, [string]$Directory, [int]$TimeoutSeconds = 60)
    $process = New-CleanProcess $Executable $ToolArguments $Directory
    try {
        $stdout = $process.StandardOutput.ReadToEndAsync()
        $stderr = $process.StandardError.ReadToEndAsync()
        $deadline = [DateTime]::UtcNow.AddSeconds($TimeoutSeconds)
        while (!$process.WaitForExit(1000)) {
            if ([DateTime]::UtcNow -ge $deadline) {
                $process.Kill()
                $process.WaitForExit()
                throw "Timed out running $Executable"
            }
        }
        $output = $stdout.GetAwaiter().GetResult()
        $errors = $stderr.GetAwaiter().GetResult()
        if ($process.ExitCode -ne 0) {
            throw "$Executable exited with $($process.ExitCode): $errors $output"
        }
        return $output
    } finally { $process.Dispose() }
}

$installer = New-Object -ComObject WindowsInstaller.Installer
$database = $installer.GetType().InvokeMember('OpenDatabase', 'InvokeMethod', $null, $installer, @($MsiPath, 0))
try {
    $properties = @{}
    Get-MsiRows $database 'SELECT `Property`, `Value` FROM `Property`' @('Name', 'Value') |
        ForEach-Object { $properties[$_.Name] = $_.Value }
    if ($properties['ProductName'] -ne 'FeatherTalk' -or
        $properties['ProductVersion'] -ne $manifest.version -or
        $properties['ProductLanguage'] -ne '2052' -or
        $properties['ALLUSERS'] -ne '1' -or
        $properties['UpgradeCode'] -ne '{EFC2A059-DB63-4C4E-93C7-177360BDE4E1}') {
        throw 'Unexpected MSI identity, language, installation scope, or upgrade code.'
    }
    $upgrades = @(Get-MsiRows $database 'SELECT `UpgradeCode`, `VersionMin`, `VersionMax`, `Attributes`, `ActionProperty` FROM `Upgrade`' @('Code', 'Minimum', 'Maximum', 'Attributes', 'ActionProperty'))
    $replacement = @($upgrades | Where-Object ActionProperty -EQ 'WIX_UPGRADE_DETECTED')
    if ($replacement.Count -ne 1 -or $replacement[0].Code -ne $properties['UpgradeCode'] -or
        $replacement[0].Minimum -ne '' -or $replacement[0].Maximum -ne $manifest.version -or
        ([int]$replacement[0].Attributes -band 512) -eq 0 -or
        ([int]$replacement[0].Attributes -band 2) -ne 0) {
        throw 'The MSI must replace older and same-version builds (VersionMaxInclusive, not OnlyDetect).'
    }
    $downgrade = @($upgrades | Where-Object ActionProperty -EQ 'WIX_DOWNGRADE_DETECTED')
    if ($downgrade.Count -ne 1 -or $downgrade[0].Code -ne $properties['UpgradeCode'] -or
        $downgrade[0].Minimum -ne $manifest.version -or $downgrade[0].Maximum -ne '' -or
        ([int]$downgrade[0].Attributes -band 2) -eq 0 -or
        ([int]$downgrade[0].Attributes -band 256) -ne 0) {
        throw 'The MSI must detect strictly newer installed versions for downgrade protection.'
    }
    $conditions = @(Get-MsiRows $database 'SELECT `Condition` FROM `LaunchCondition`' @('Condition'))
    if (@($conditions | Where-Object Condition -EQ 'NOT WIX_DOWNGRADE_DETECTED').Count -ne 1) {
        throw 'The MSI must block installation over a newer installed version.'
    }
    $shortcuts = @(Get-MsiRows $database 'SELECT `Name`, `Directory_`, `Target`, `WkDir` FROM `Shortcut`' @('Name', 'Directory', 'Target', 'WorkingDirectory'))
    if ($shortcuts.Count -ne 1 -or $shortcuts[0].Directory -ne 'ProgramMenuFolder' -or
        $shortcuts[0].Target -ne 'Complete' -or $shortcuts[0].WorkingDirectory -ne 'INSTALLFOLDER') {
        throw 'The MSI must contain the advertised Start menu shortcut for the app.'
    }
    $directories = @{}
    Get-MsiRows $database 'SELECT `Directory`, `Directory_Parent`, `DefaultDir` FROM `Directory`' @('Id', 'Parent', 'Name') |
        ForEach-Object { $directories[$_.Id] = $_ }
    $components = @{}
    Get-MsiRows $database 'SELECT `Component`, `Directory_` FROM `Component`' @('Id', 'Directory') |
        ForEach-Object { $components[$_.Id] = $_.Directory }
    $files = @(Get-MsiRows $database 'SELECT `FileName`, `FileSize`, `Component_` FROM `File`' @('Name', 'Bytes', 'Component'))
    $msiFiles = @{}
    foreach ($file in $files) {
        $directory = Get-RelativeMsiDirectory $components[$file.Component] $directories
        $name = ($file.Name -split '\|')[-1]
        $relative = if ($directory) { "$directory/$name" } else { $name }
        if ($msiFiles.ContainsKey($relative)) { throw "Duplicate installed payload path: $relative" }
        $msiFiles[$relative] = [long]$file.Bytes
    }
    if ($msiFiles.Count -ne $manifest.files.Count) { throw 'MSI and payload manifest file counts differ.' }
    foreach ($file in $manifest.files) {
        if (!$msiFiles.ContainsKey($file.name) -or $msiFiles[$file.name] -ne $file.bytes) {
            throw "MSI file table mismatch: $($file.name)"
        }
    }
    $summary = $database.GetType().InvokeMember('SummaryInformation', 'GetProperty', $null, $database, @(0))
    try {
        $template = $summary.GetType().InvokeMember('Property', 'GetProperty', $null, $summary, @(7))
        if ($template -ne 'x64;2052') { throw "Unexpected MSI platform: $template" }
    } finally { [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($summary) }
} finally {
    [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($database)
    [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($installer)
}
Write-Host "MSI metadata verified: $($manifest.version), x64, Chinese UI, $($manifest.files.Count) files."

$verificationDirectory = Join-Path $rustRoot ("target/installer/verify-" + [Guid]::NewGuid().ToString('N'))
$imageDirectory = Join-Path $verificationDirectory 'Extracted image'
$workingDirectory = Join-Path $verificationDirectory 'Working directory'
New-Item -ItemType Directory -Path $imageDirectory, $workingDirectory -Force | Out-Null
$installLog = Join-Path $verificationDirectory 'extract.log'
$installArguments = '/a ' + (ConvertTo-NativeArgument $MsiPath) + ' /qn TARGETDIR=' +
    (ConvertTo-NativeArgument $imageDirectory) + ' /L*v ' + (ConvertTo-NativeArgument $installLog)
$extraction = Start-Process -FilePath "$env:SystemRoot/System32/msiexec.exe" -ArgumentList $installArguments -WindowStyle Hidden -PassThru
try {
    if (!$extraction.WaitForExit(60000)) { throw "MSI extraction is still running; see $installLog" }
    if ($extraction.ExitCode -ne 0) { throw "MSI extraction failed ($($extraction.ExitCode)); see $installLog" }
} finally { $extraction.Dispose() }
$apps = @(Get-ChildItem -LiteralPath $imageDirectory -Filter 'feathertalk-app.exe' -File -Recurse)
if ($apps.Count -ne 1) { throw 'Expected exactly one app in the extracted image.' }
$installedDirectory = $apps[0].DirectoryName
foreach ($file in $manifest.files) {
    $path = Join-Path $installedDirectory $file.name
    if ((Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash -ne $file.sha256) {
        throw "Extracted file checksum mismatch: $($file.name)"
    }
}
$reader = [IO.BinaryReader]::new([IO.File]::OpenRead($apps[0].FullName))
try {
    $reader.BaseStream.Position = 0x3c
    $header = $reader.ReadInt32()
    $reader.BaseStream.Position = $header + 92
    if ($reader.ReadUInt16() -ne 2) { throw 'The desktop release must use the Windows GUI subsystem.' }
} finally { $reader.Dispose() }
Write-Host 'Administrative extraction, payload hashes, and desktop GUI subsystem verified.'

foreach ($definition in Get-BundledModelDefinitions) {
    $packageDirectory = Join-Path $installedDirectory "models/$($definition.directory)"
    $model = Assert-BundledModelPackage $packageDirectory $definition
    $declared = @($manifest.models | Where-Object directory -EQ $model.directory)
    if ($declared.Count -ne 1 -or $declared[0].model_kind -ne $model.model_kind -or
        $declared[0].source_file -ne $model.source_file -or $declared[0].source_sha256 -ne $model.source_sha256 -or
        $declared[0].model_sha256 -ne $model.model_sha256) {
        throw "Model provenance differs from the payload manifest: $($model.directory)"
    }
}
Write-Host 'All four model packages match their source checkpoints and weight hashes.'

$cli = Join-Path $installedDirectory 'feathertalk.exe'
$cliVersion = (Invoke-CleanProcess $cli @('--version') $workingDirectory).Trim()
if ($cliVersion -ne "feathertalk $($manifest.version)") { throw "Unexpected CLI version: $cliVersion" }
$ready = Invoke-CleanProcess $cli @('--json', 'capabilities') $workingDirectory | ConvertFrom-Json
if ($ready.frame -ne 'ready' -or !$ready.data.capabilities.ffmpeg -or $ready.data.worker_version -ne $manifest.version) {
    throw 'The extracted CLI did not discover its worker and bundled media tools.'
}
if (!$ready.data.capabilities.training) { throw 'The worker did not discover the bundled VGG19 training model.' }
foreach ($command in @('extract_frames', 'extract_features', 'lock_asset_package', 'train')) {
    if ($ready.data.supported_commands -notcontains $command) {
        throw "The worker did not discover the bundled models required for $command."
    }
}
$fixture = Join-Path $workingDirectory 'input.mp4'
$ffmpeg = Join-Path $installedDirectory 'ffmpeg.exe'
Invoke-CleanProcess $ffmpeg @('-hide_banner', '-loglevel', 'error', '-f', 'lavfi', '-i', 'color=c=black:s=160x160:r=25:d=1',
    '-f', 'lavfi', '-i', 'sine=frequency=440:sample_rate=16000:duration=1', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
    '-c:a', 'aac', '-shortest', $fixture) $workingDirectory | Out-Null
Invoke-CleanProcess $cli @('probe-media', $fixture) $workingDirectory | Out-Null
$normalized = Join-Path $workingDirectory 'normalized'
Invoke-CleanProcess $cli @('normalize-media', $fixture, $normalized) $workingDirectory | Out-Null
$mediaFiles = @(Get-ChildItem -LiteralPath $normalized -Recurse -File | Where-Object Extension -In @('.mp4', '.wav'))
if ($mediaFiles.Count -ne 2 -or @($mediaFiles | Where-Object Length -EQ 0).Count) {
    throw 'Normalization did not publish a nonempty video and audio pair.'
}
Write-Host 'CLI version, worker discovery, bundled FFmpeg, media probing, and normalization passed with a clean tool environment.'

$hubertDirectory = Join-Path $installedDirectory 'models/feather_hubert'
$inspection = Invoke-CleanProcess $cli @('inspect-model', $hubertDirectory) $workingDirectory | ConvertFrom-Json
if (!$inspection.compatible -or $inspection.model_kind -ne 'feather_hubert') {
    throw 'The bundled FeatherHuBERT model package is incompatible with the installed worker.'
}
$project = Join-Path $workingDirectory 'Model smoke project'
$assets = Join-Path $project 'assets'
New-Item -ItemType Directory -Path $assets -Force | Out-Null
[IO.File]::WriteAllText((Join-Path $project 'project.json'), '{}', [Text.UTF8Encoding]::new($false))
$video = Join-Path $assets 'video_25fps.mp4'
$faceFixture = Join-Path $rustRoot 'crates/feathertalk-frame-adapters/tests/fixtures/demo_frame_v1/frame.jpg'
Invoke-CleanProcess $ffmpeg @('-hide_banner', '-loglevel', 'error', '-loop', '1', '-framerate', '25', '-i', $faceFixture,
    '-frames:v', '1', '-an', '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p', $video) $workingDirectory | Out-Null
$frames = Invoke-CleanProcess $cli @('extract-frames', $project, $video) $workingDirectory | ConvertFrom-Json
$quality = Get-Content -LiteralPath (Join-Path $assets 'quality.json') -Raw -Encoding UTF8 | ConvertFrom-Json
if ($frames.frame_count -ne 1 -or $quality.accepted_count -ne 1 -or
    !(Test-Path -LiteralPath (Join-Path $assets 'landmarks/000000.lms') -PathType Leaf)) {
    throw 'Bundled SCRFD/PFLD did not produce a valid face and landmark result.'
}
$audio = Join-Path $assets 'audio_16k_mono.wav'
Invoke-CleanProcess $ffmpeg @('-hide_banner', '-loglevel', 'error', '-f', 'lavfi', '-i',
    'sine=frequency=440:sample_rate=16000:duration=1', '-ac', '1', '-c:a', 'pcm_s16le', $audio) $workingDirectory | Out-Null
$features = Invoke-CleanProcess $cli @('extract-features', $project, $audio) $workingDirectory | ConvertFrom-Json
$hubertManifest = Get-Content -LiteralPath (Join-Path $hubertDirectory 'manifest.json') -Raw -Encoding UTF8 | ConvertFrom-Json
$featureFile = Get-Item -LiteralPath (Join-Path $assets 'features/feather_hubert.f32')
if ($features.tokens -ne 48 -or $features.dims -ne 1024 -or $featureFile.Length -ne 196652 -or
    $features.model_sha256 -ne $hubertManifest.model.sha256) {
    throw 'Bundled FeatherHuBERT did not produce the expected one-second audio features.'
}
Write-Host 'Bundled SCRFD/PFLD face and landmark inference and FeatherHuBERT audio inference passed on CPU with a clean tool environment.'

$trainingProject = Join-Path $workingDirectory 'Training smoke project'
$trainingAssets = Join-Path $trainingProject 'assets'
New-Item -ItemType Directory -Path (Join-Path $trainingAssets 'frames'), (Join-Path $trainingAssets 'landmarks') -Force | Out-Null
$projectManifest = [ordered]@{
    schema_version = 1; project_id = 'installer-smoke'; display_name = 'Installer model verification'
    asset_package = 'assets/assets.json'; default_model = 'original_unet'; task_history = @()
}
[IO.File]::WriteAllText((Join-Path $trainingProject 'project.json'), ($projectManifest | ConvertTo-Json), [Text.UTF8Encoding]::new($false))
foreach ($name in @('video_25fps.mp4', 'quality.json', 'frames/000000.jpg', 'landmarks/000000.lms')) {
    Copy-Item -LiteralPath (Join-Path $assets $name) -Destination (Join-Path $trainingAssets $name)
}
$trainingAudio = Join-Path $trainingAssets 'audio_16k_mono.wav'
Invoke-CleanProcess $ffmpeg @('-hide_banner', '-loglevel', 'error', '-f', 'lavfi', '-i',
    'sine=frequency=440:sample_rate=16000:duration=0.05', '-ac', '1', '-c:a', 'pcm_s16le', $trainingAudio) $workingDirectory | Out-Null
$trainingFeatures = Invoke-CleanProcess $cli @('extract-features', $trainingProject, $trainingAudio) $workingDirectory | ConvertFrom-Json
if ($trainingFeatures.tokens -ne 2 -or $trainingFeatures.frame_count -ne 1) {
    throw 'The training fixture must contain one frame and two audio feature tokens.'
}
$locked = Invoke-CleanProcess $cli @('lock-asset-package', $trainingProject) $workingDirectory | ConvertFrom-Json
if ($locked.frame_count -ne 1) { throw 'The training fixture was not locked with exactly one frame.' }
Write-Host 'Training one CPU step with the bundled VGG19 perceptual model...'
$trained = Invoke-CleanProcess $cli @('train', $trainingProject, '--mode', 'baseline', '--epochs', '1') $workingDirectory 600 | ConvertFrom-Json
$loss = [double]$trained.total_loss
if ($trained.epochs_completed -ne 1 -or $trained.global_step -ne 1 -or $trained.checkpoints_written -ne 1 -or
    $trained.backend -ne 'ndarray-cpu' -or [double]::IsNaN($loss) -or [double]::IsInfinity($loss) -or $loss -lt 0 -or
    !(Test-Path -LiteralPath (Join-Path $trainingProject 'models/unet/checkpoint-00000001/manifest.json') -PathType Leaf)) {
    throw 'Training with the bundled VGG19 package did not publish a valid CPU checkpoint.'
}
Write-Host 'Bundled VGG19 perceptual loss passed a complete CPU training step and checkpoint save with a clean tool environment.'

if (!$SkipGuiSmoke) {
    $app = New-CleanProcess $apps[0].FullName @() $workingDirectory
    try {
        $stdout = $app.StandardOutput.ReadToEndAsync()
        $stderr = $app.StandardError.ReadToEndAsync()
        $deadline = [DateTime]::UtcNow.AddSeconds(20)
        do {
            Start-Sleep -Milliseconds 250
            $app.Refresh()
        } while (!$app.HasExited -and $app.MainWindowHandle -eq [IntPtr]::Zero -and [DateTime]::UtcNow -lt $deadline)
        if ($app.HasExited) { throw "Desktop startup failed: $($stderr.GetAwaiter().GetResult())" }
        if ($app.MainWindowHandle -eq [IntPtr]::Zero -or !$app.Responding) { throw 'Desktop startup did not produce a responsive window.' }
        Write-Host "Desktop startup passed: $($app.MainWindowTitle)"
    } finally {
        if (!$app.HasExited) { $app.Kill(); $app.WaitForExit() }
        $app.Dispose()
    }
}
Write-Host "Verification passed. Evidence: $verificationDirectory"
