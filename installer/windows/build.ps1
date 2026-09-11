#Requires -Version 5.1
[CmdletBinding()]
param(
    [string]$FfmpegDirectory,
    [string]$SourceRepositoryDirectory,
    [string]$VCRuntimeDirectory,
    [string]$ScrfdModelDirectory,
    [string]$PfldModelDirectory,
    [string]$HubertModelDirectory,
    [string]$Vgg19ModelDirectory,
    [string]$OutputDirectory,
    [switch]$SkipBuild
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
Set-StrictMode -Version Latest
. (Join-Path $PSScriptRoot 'models.ps1')

if ([Environment]::OSVersion.Platform -ne [PlatformID]::Win32NT -or ![Environment]::Is64BitProcess) {
    throw 'Build the installer in a 64-bit PowerShell process on Windows.'
}

$rustRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$repositoryRoot = if ($SourceRepositoryDirectory) {
    (Resolve-Path -LiteralPath $SourceRepositoryDirectory).Path
} else {
    [IO.Path]::GetFullPath((Join-Path $rustRoot '..'))
}
$licensePath = Join-Path $repositoryRoot 'LICENSE'
if (!(Test-Path -LiteralPath $licensePath -PathType Leaf)) {
    throw "Project LICENSE not found at $licensePath. Pass -SourceRepositoryDirectory with the original FeatherTalk repository."
}
$utf8 = [Text.UTF8Encoding]::new($false)
$cargo = (Get-Command cargo -CommandType Application).Source
$dotnet = (Get-Command dotnet -CommandType Application).Source

function Invoke-Checked {
    param([string]$Executable, [string[]]$ToolArguments)
    # rustup selects rust-toolchain.toml from the working directory, not --manifest-path.
    Push-Location -LiteralPath $rustRoot
    try {
        & $Executable @ToolArguments | Out-Host
        if ($LASTEXITCODE -ne 0) {
            throw "$Executable failed with exit code $LASTEXITCODE."
        }
    } finally { Pop-Location }
}

function Get-CargoMetadata {
    param([string]$Manifest)
    Push-Location -LiteralPath $rustRoot
    try {
        $json = & $cargo metadata --manifest-path $Manifest --locked --no-deps --format-version 1
        if ($LASTEXITCODE -ne 0) { throw "Cargo metadata failed for $Manifest." }
        return (($json -join "`n") | ConvertFrom-Json)
    } finally { Pop-Location }
}

function Invoke-CargoReleaseBuild {
    param([string[]]$ToolArguments)
    Push-Location -LiteralPath $rustRoot
    try {
        $messages = & $cargo @ToolArguments --message-format json-render-diagnostics
        if ($LASTEXITCODE -ne 0) { throw "Cargo release build failed with exit code $LASTEXITCODE." }
        $artifacts = @{}
        foreach ($line in $messages) {
            $message = $line | ConvertFrom-Json
            if ($message.reason -eq 'compiler-artifact' -and $message.executable) {
                if ($artifacts.ContainsKey($message.target.name)) {
                    throw "Multiple target architectures produced $($message.target.name); build one x64 target."
                }
                $artifacts[$message.target.name] = [string]$message.executable
            }
        }
        return $artifacts
    } finally { Pop-Location }
}

function Assert-X64Binary {
    param([string]$Path)
    $reader = [IO.BinaryReader]::new([IO.File]::OpenRead($Path))
    try {
        if ($reader.ReadUInt16() -ne 0x5a4d) { throw "Not a PE executable: $Path" }
        $reader.BaseStream.Position = 0x3c
        $header = $reader.ReadInt32()
        $reader.BaseStream.Position = $header
        if ($reader.ReadUInt32() -ne 0x4550 -or $reader.ReadUInt16() -ne 0x8664) {
            throw "Not a Windows x64 binary: $Path"
        }
    } finally { $reader.Dispose() }
}

function Get-WixPackage {
    param([string]$Package, [string]$CacheName, [string]$Sha256, [string]$EntryPoint)
    $version = '5.0.2'
    $cache = Join-Path $rustRoot "target/wix-tools/$version"
    New-Item -ItemType Directory -Path $cache -Force | Out-Null
    $archive = Join-Path $cache "$CacheName.$version.nupkg"
    if (!(Test-Path -LiteralPath $archive -PathType Leaf)) {
        Write-Host "Downloading $Package $version from NuGet..."
        Invoke-WebRequest -UseBasicParsing -Uri "https://api.nuget.org/v3-flatcontainer/$Package/$version/$Package.$version.nupkg" -OutFile $archive
    }
    if ((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash -ne $Sha256) {
        throw "WiX package checksum mismatch: $archive"
    }
    $expanded = Join-Path $cache $CacheName
    if (!(Test-Path -LiteralPath $expanded)) {
        [IO.Compression.ZipFile]::ExtractToDirectory($archive, $expanded)
    }
    $entry = Join-Path $expanded $EntryPoint
    if (!(Test-Path -LiteralPath $entry -PathType Leaf)) {
        throw "Incomplete WiX cache; restore this package directory: $expanded"
    }
    return $entry
}

$workspaceManifest = Join-Path $rustRoot 'Cargo.toml'
$appManifest = Join-Path $rustRoot 'crates/feathertalk-app/Cargo.toml'
$workspace = Get-CargoMetadata $workspaceManifest
$application = Get-CargoMetadata $appManifest
$workerPackage = $workspace.packages | Where-Object name -eq 'feathertalk-worker'
$appPackage = $application.packages | Where-Object name -eq 'feathertalk-app'
$version = [string]$workerPackage.version
if ($version -ne $appPackage.version -or $version -notmatch '^\d+\.\d+\.\d+$') {
    throw 'The app and worker need the same three-part release version.'
}
$msiVersion = [Version]$version
if ($msiVersion.Major -gt 255 -or $msiVersion.Minor -gt 255 -or $msiVersion.Build -gt 65535) {
    throw "Version $version exceeds Windows Installer version limits."
}

if (!$FfmpegDirectory) {
    $FfmpegDirectory = Split-Path -Parent (Get-Command ffmpeg -CommandType Application).Source
}
$FfmpegDirectory = (Resolve-Path -LiteralPath $FfmpegDirectory).Path
$ffmpegBin = $FfmpegDirectory
if (Test-Path -LiteralPath (Join-Path $FfmpegDirectory 'bin/ffmpeg.exe')) {
    $ffmpegBin = Join-Path $FfmpegDirectory 'bin'
}
$ffmpegRoot = $ffmpegBin
if (!(Test-Path -LiteralPath (Join-Path $ffmpegRoot 'LICENSE'))) {
    $ffmpegRoot = Split-Path -Parent $ffmpegBin
}
foreach ($file in @('ffmpeg.exe', 'ffprobe.exe')) {
    $path = Join-Path $ffmpegBin $file
    if (!(Test-Path -LiteralPath $path -PathType Leaf)) { throw "Missing $path" }
    Assert-X64Binary $path
}
foreach ($file in @('LICENSE', 'README.txt')) {
    if (!(Test-Path -LiteralPath (Join-Path $ffmpegRoot $file) -PathType Leaf)) {
        throw "The FFmpeg distribution must include $file for redistribution."
    }
}

if (!$VCRuntimeDirectory) {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
    if (!(Test-Path -LiteralPath $vswhere)) { throw 'Pass -VCRuntimeDirectory with the x64 MSVC CRT redistributable directory.' }
    $visualStudio = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if (!$visualStudio) { throw 'Visual Studio C++ Build Tools were not found.' }
    $redist = Get-ChildItem -LiteralPath (Join-Path $visualStudio 'VC/Redist/MSVC') -Directory |
        Where-Object Name -Match '^\d+\.\d+\.\d+$' |
        Sort-Object { [Version]$_.Name } -Descending | Select-Object -First 1
    if (!$redist) { throw 'The MSVC redistributable directory was not found.' }
    $VCRuntimeDirectory = Join-Path $redist.FullName 'x64/Microsoft.VC143.CRT'
}
$VCRuntimeDirectory = (Resolve-Path -LiteralPath $VCRuntimeDirectory).Path
if (!(Test-Path -LiteralPath (Join-Path $VCRuntimeDirectory 'vcruntime140.dll') -PathType Leaf)) {
    throw 'The runtime directory must contain the redistributable vcruntime140.dll.'
}

Add-Type -AssemblyName System.IO.Compression.FileSystem
$wix = Get-WixPackage 'wix' 'wix' 'F30EF0C74E2A986126539C5780BE93AC24E8136EAF723B1937B26272703AE173' 'tools/net6.0/any/wix.dll'
$wixUi = Get-WixPackage 'wixtoolset.ui.wixext' 'ui' '5EF2C707614B9F70B6BBADD2D4ABCB4124EFEE215E9B16BFBC80113079A604C7' 'wixext5/WixToolset.UI.wixext.dll'
Invoke-Checked $dotnet @($wix, '--version')

if (!$SkipBuild) {
    Write-Host 'Building release worker and CLI...'
    $workerArtifacts = Invoke-CargoReleaseBuild @('build', '--manifest-path', $workspaceManifest, '--release', '--locked', '-p', 'feathertalk-worker', '-p', 'feathertalk-cli', '--bin', 'feathertalk-worker', '--bin', 'feathertalk')
    Write-Host 'Building release desktop app...'
    $appArtifacts = Invoke-CargoReleaseBuild @('build', '--manifest-path', $appManifest, '--release', '--locked', '--bin', 'feathertalk-app')
    # Use the artifacts Cargo actually produced, including an explicitly configured target.
    $appExecutable = $appArtifacts['feathertalk-app']
    $workerExecutable = $workerArtifacts['feathertalk-worker']
    $cliExecutable = $workerArtifacts['feathertalk']
    if (!$appExecutable -or !$workerExecutable -or !$cliExecutable) {
        throw 'Cargo did not report all three production executables.'
    }
} else {
    # Explicit native-release reuse; normal builds always use Cargo's artifact paths above.
    $appExecutable = Join-Path $application.target_directory 'release/feathertalk-app.exe'
    $workerExecutable = Join-Path $workspace.target_directory 'release/feathertalk-worker.exe'
    $cliExecutable = Join-Path $workspace.target_directory 'release/feathertalk.exe'
}

# A fresh stage prevents stale artifacts from entering the MSI and needs no deletion.
$buildDirectory = Join-Path $rustRoot ("target/installer/build-" + [Guid]::NewGuid().ToString('N'))
$stage = Join-Path $buildDirectory 'payload'
New-Item -ItemType Directory -Path $stage -Force | Out-Null
$binaries = @(
    $appExecutable,
    $workerExecutable,
    $cliExecutable,
    (Join-Path $ffmpegBin 'ffmpeg.exe'),
    (Join-Path $ffmpegBin 'ffprobe.exe')
)
$binaries += @(Get-ChildItem -LiteralPath $VCRuntimeDirectory -Filter '*.dll' -File | Select-Object -ExpandProperty FullName)
$binaries += @(Get-ChildItem -LiteralPath $ffmpegBin -Filter '*.dll' -File | Select-Object -ExpandProperty FullName)
foreach ($binary in $binaries) {
    Assert-X64Binary $binary
    $destination = Join-Path $stage ([IO.Path]::GetFileName($binary))
    if (Test-Path -LiteralPath $destination) { throw "Duplicate payload filename: $destination" }
    Copy-Item -LiteralPath $binary -Destination $destination
}

if (!$ScrfdModelDirectory) { $ScrfdModelDirectory = Join-Path $rustRoot 'crates/feathertalk-scrfd/artifacts/scrfd_2_5g' }
if (!$PfldModelDirectory) { $PfldModelDirectory = Join-Path $rustRoot 'crates/feathertalk-pfld/artifacts/pfld_ghost_one' }
if (!$Vgg19ModelDirectory) { $Vgg19ModelDirectory = Join-Path $rustRoot 'target/installer/models/vgg19' }
$modelDefinitions = @(Get-BundledModelDefinitions)
$modelSources = @{
    scrfd_2_5g = $ScrfdModelDirectory; pfld_ghost_one = $PfldModelDirectory
    feather_hubert = $HubertModelDirectory; vgg19 = $Vgg19ModelDirectory
}
$modelRecords = @()
foreach ($definition in $modelDefinitions) {
    $destination = Join-Path $stage "models/$($definition.directory)"
    if ($definition.directory -eq 'feather_hubert' -and !$HubertModelDirectory) {
        # Import through the production Rust worker. Keep conversion inputs outside the payload.
        $checkpoint = Join-Path $repositoryRoot "demo/kanghui_training_video_featherhubert_188_latest/$($definition.source_file)"
        if ((Get-FileHash -LiteralPath $checkpoint -Algorithm SHA256).Hash -ne $definition.source_sha256) {
            throw "The FeatherHuBERT checkpoint does not match $($definition.source_file)."
        }
        $conversionInput = Join-Path $buildDirectory 'conversion-input'
        New-Item -ItemType Directory -Path $conversionInput, (Split-Path -Parent $destination) -Force | Out-Null
        $stagedCheckpoint = Join-Path $conversionInput $definition.source_file
        Copy-Item -LiteralPath $checkpoint -Destination $stagedCheckpoint
        Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'feather-hubert-licenses.json') -Destination (Join-Path $conversionInput 'LICENSES.json')
        Write-Host 'Converting the FeatherHuBERT checkpoint with the Rust worker...'
        Invoke-Checked $cliExecutable @('--worker', $workerExecutable, '--backend', 'cpu', '--adapter', 'cpu-0',
            'import-legacy-model', $stagedCheckpoint, 'feather-hubert', $destination)
    } else {
        $sourceDirectory = (Resolve-Path -LiteralPath $modelSources[$definition.directory]).Path
        New-Item -ItemType Directory -Path $destination -Force | Out-Null
        foreach ($name in $definition.files) {
            Copy-Item -LiteralPath (Join-Path $sourceDirectory $name) -Destination (Join-Path $destination $name)
        }
    }
    $modelRecords += Assert-BundledModelPackage $destination $definition
}
Write-Host 'Bundled SCRFD, PFLD, FeatherHuBERT, and VGG19 packages passed provenance and weight checks.'

Copy-Item -LiteralPath $licensePath -Destination (Join-Path $stage 'LICENSE.txt')
Copy-Item -LiteralPath (Join-Path $ffmpegRoot 'LICENSE') -Destination (Join-Path $stage 'FFmpeg-LICENSE.txt')
Copy-Item -LiteralPath (Join-Path $ffmpegRoot 'README.txt') -Destination (Join-Path $stage 'FFmpeg-README.txt')
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'README.md') -Destination (Join-Path $stage 'README.txt')
$cubekNotices = Join-Path $stage 'licenses/cubek-matmul'
New-Item -ItemType Directory -Path $cubekNotices -Force | Out-Null
foreach ($name in @('LICENSE-MIT', 'LICENSE-APACHE', 'FEATHERTALK-PATCH.md')) {
    Copy-Item -LiteralPath (Join-Path $rustRoot "vendor/cubek-matmul/$name") -Destination (Join-Path $cubekNotices $name)
}
$notices = @'
FeatherTalk is licensed under Apache-2.0; see LICENSE.txt.
Project and source: https://github.com/anliyuan/FeatherTalk

The worker includes cubek-matmul 0.2.0 (MIT OR Apache-2.0) with a local
double-buffered K bounds correction. Upstream licenses, source provenance,
and the modification notice are included under licenses/cubek-matmul.
Cubek project: https://github.com/tracel-ai/cubek

FFmpeg and FFprobe are separate executables, distributed without modification.
Their license, source reference, version and build configuration are provided in
FFmpeg-LICENSE.txt and FFmpeg-README.txt. These licenses also apply to the
components incorporated by that FFmpeg distribution.
FFmpeg project: https://ffmpeg.org/

Microsoft Visual C++ runtime DLLs are copied from the x64 redistributable files
of Visual Studio C++ Build Tools. Copyright Microsoft Corporation.
https://learn.microsoft.com/visualstudio/releases/2022/redistribution

Bundled models are under models/scrfd_2_5g, models/pfld_ghost_one,
models/feather_hubert, and models/vgg19. Their manifests retain the original
checkpoint identities and license metadata. FeatherHuBERT and VGG19 also carry
LICENSES.json provenance records. VGG19 contains the conv3_3 feature extractor
used by perceptual loss. The application's Apache-2.0 license does not relicense
model weights.
'@
[IO.File]::WriteAllText((Join-Path $stage 'THIRD-PARTY-NOTICES.txt'), $notices, $utf8)

$license = [IO.File]::ReadAllText($licensePath)
$rtf = '{\rtf1\ansi\deff0{\fonttbl{\f0 Segoe UI;}}\f0\fs18 ' +
    $license.Replace('\', '\\').Replace('{', '\{').Replace('}', '\}').Replace("`r", '').Replace("`n", "\par`n") + '}'
$licenseRtf = Join-Path $buildDirectory 'License.rtf'
[IO.File]::WriteAllText($licenseRtf, $rtf, [Text.Encoding]::ASCII)

$payloadSource = Join-Path $buildDirectory 'Payload.wxs'
$xml = [Text.StringBuilder]::new()
[void]$xml.AppendLine('<Wix xmlns="http://wixtoolset.org/schemas/v4/wxs"><Fragment>')
$sha = [Security.Cryptography.SHA256]::Create()
try {
    $directoryIds = @{ $stage = 'INSTALLFOLDER' }
    foreach ($directory in Get-ChildItem -LiteralPath $stage -Directory -Recurse | Sort-Object FullName) {
        $relative = $directory.FullName.Substring($stage.Length + 1).Replace('\', '/')
        $id = 'D' + ([BitConverter]::ToString($sha.ComputeHash($utf8.GetBytes($relative)))).Replace('-', '').Substring(0, 24)
        $parentId = $directoryIds[$directory.Parent.FullName]
        $name = [Security.SecurityElement]::Escape($directory.Name)
        [void]$xml.AppendLine("<DirectoryRef Id=`"$parentId`"><Directory Id=`"$id`" Name=`"$name`" /></DirectoryRef>")
        $directoryIds[$directory.FullName] = $id
    }
    [void]$xml.AppendLine('<ComponentGroup Id="PayloadFiles">')
    foreach ($file in Get-ChildItem -LiteralPath $stage -File -Recurse | Sort-Object FullName) {
        $relative = $file.FullName.Substring($stage.Length + 1).Replace('\', '/')
        if ($relative -eq 'feathertalk-app.exe') { continue }
        $id = ([BitConverter]::ToString($sha.ComputeHash($utf8.GetBytes($relative)))).Replace('-', '').Substring(0, 24)
        $directoryId = $directoryIds[$file.DirectoryName]
        $source = [Security.SecurityElement]::Escape($file.FullName)
        [void]$xml.AppendLine("<Component Id=`"C$id`" Guid=`"*`" Directory=`"$directoryId`"><File Id=`"F$id`" Source=`"$source`" KeyPath=`"yes`" /></Component>")
    }
} finally { $sha.Dispose() }
[void]$xml.AppendLine('</ComponentGroup></Fragment></Wix>')
[IO.File]::WriteAllText($payloadSource, $xml.ToString(), $utf8)

if (!$OutputDirectory) { $OutputDirectory = Join-Path $rustRoot 'dist' }
$OutputDirectory = [IO.Path]::GetFullPath($OutputDirectory)
New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null
$msi = Join-Path $OutputDirectory "FeatherTalk-$version-x64.msi"
Write-Host "Packaging $msi ..."
Invoke-Checked $dotnet @($wix, 'build', '-arch', 'x64', '-culture', 'zh-CN', '-ext', $wixUi,
    '-d', "ProductVersion=$version", '-d', "SourceDir=$stage", '-d', "LicenseRtf=$licenseRtf",
    '-pdbtype', 'none', '-intermediatefolder', (Join-Path $buildDirectory 'wix'),
    (Join-Path $PSScriptRoot 'Package.wxs'), $payloadSource, '-out', $msi)
Invoke-Checked $dotnet @($wix, 'msi', 'validate', $msi)

$payload = @(Get-ChildItem -LiteralPath $stage -File -Recurse | Sort-Object FullName | ForEach-Object {
    $relative = $_.FullName.Substring($stage.Length + 1).Replace('\', '/')
    [ordered]@{ name = $relative; bytes = $_.Length; sha256 = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant() }
})
$manifest = [ordered]@{ product = 'FeatherTalk'; version = $version; architecture = 'x64'; wix = '5.0.2'; models = $modelRecords; files = $payload }
[IO.File]::WriteAllText((Join-Path $OutputDirectory "FeatherTalk-$version-x64.payload.json"), ($manifest | ConvertTo-Json -Depth 5), $utf8)
$checksum = (Get-FileHash -LiteralPath $msi -Algorithm SHA256).Hash.ToLowerInvariant()
[IO.File]::WriteAllText("$msi.sha256", "$checksum  $([IO.Path]::GetFileName($msi))`n", $utf8)
Write-Host "Created $msi ($([math]::Round((Get-Item -LiteralPath $msi).Length / 1MB, 1)) MiB)"
Write-Host "SHA256 $checksum"
