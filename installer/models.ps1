# Shared model identities and package checks for building and verifying the MSI.
function Get-BundledModelDefinitions {
    @(
        [pscustomobject]@{
            directory = 'scrfd_2_5g'; kind = 'scrfd_2.5g_kps'; source_file = 'scrfd_2.5g_kps.onnx'
            source_sha256 = '32d20c77b9e2dc1d07e94c2ab9d25bdd5cd05eddbe0b46e7b38e7a1eca22e99a'
            files = @('manifest.json', 'model.safetensors')
        },
        [pscustomobject]@{
            directory = 'pfld_ghost_one'; kind = 'pfld_ghost_one'; source_file = 'checkpoint_epoch_335.pth.tar'
            source_sha256 = 'bada866661ad5fa1080a085f51fe9c016c69958c406951afa4afc7840f856de0'
            files = @('manifest.json', 'model.safetensors')
        },
        [pscustomobject]@{
            directory = 'feather_hubert'; kind = 'feather_hubert'; source_file = 'feather_hubert_188_latest_99.pth'
            source_sha256 = '58df96af118d75d7f69da441e1f3960096f28dda637a4e8f4265f108d27aeb52'
            files = @('manifest.json', 'model.safetensors', 'LICENSES.json')
        },
        [pscustomobject]@{
            directory = 'vgg19'; kind = 'vgg19-conv3-3'; source_file = 'vgg19-dcbb9e9d.pth'
            source_sha256 = 'dcbb9e9dad569fff7a846263a77324fc34978fea2bfb039c012d710e1776ae44'
            source_url = 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'
            files = @('manifest.json', 'model.safetensors', 'LICENSES.json')
        }
    )
}

function Assert-BundledModelPackage {
    param([string]$Directory, $Definition)
    foreach ($name in $Definition.files) {
        if (!(Test-Path -LiteralPath (Join-Path $Directory $name) -PathType Leaf)) {
            throw "Incomplete $($Definition.kind) model package: missing $name in $Directory"
        }
    }
    $package = Get-Content -LiteralPath (Join-Path $Directory 'manifest.json') -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($Definition.directory -eq 'scrfd_2_5g') {
        $kind = $package.model_kind
        $weights = $package.weights
        $sourceMatches = $package.source.file_name -eq $Definition.source_file
    } elseif ($Definition.directory -eq 'vgg19') {
        $kind = $package.model_kind
        $weights = $package.model
        # The VGG19 runtime format identifies its official source by URL and weight ID.
        $sourceMatches = $package.source.framework -eq 'torchvision' -and
            $package.source.weight_id -eq 'VGG19_Weights.IMAGENET1K_V1' -and
            $package.source.url -eq $Definition.source_url
        if ($package.architecture_version -ne 'torchvision-vgg19-conv3-3-v1' -or
            $package.output_layer -ne 'features.14' -or $package.tensor_count -ne 14 -or
            $package.total_elements -ne 1735488) {
            throw "Unexpected VGG19 perceptual feature layout in $Directory"
        }
    } else {
        $kind = $package.model_type
        $weights = $package.model
        $sourceMatches = $package.source.file_name -eq $Definition.source_file
    }
    if ($kind -ne $Definition.kind -or !$sourceMatches -or
        $package.source.sha256 -ne $Definition.source_sha256) {
        throw "Unexpected model kind or source checkpoint in $Directory"
    }
    $weightsHash = (Get-FileHash -LiteralPath (Join-Path $Directory 'model.safetensors') -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($weights.file_name -ne 'model.safetensors' -or $weights.sha256 -ne $weightsHash) {
        throw "Model weight checksum mismatch in $Directory"
    }
    if ($Definition.files -contains 'LICENSES.json') {
        $licenseHash = (Get-FileHash -LiteralPath (Join-Path $Directory 'LICENSES.json') -Algorithm SHA256).Hash
        if ($package.licenses.file_name -ne 'LICENSES.json' -or $package.licenses.sha256 -ne $licenseHash) {
            throw "Model license checksum mismatch in $Directory"
        }
    }
    [pscustomobject][ordered]@{
        directory = "models/$($Definition.directory)"
        model_kind = $kind
        source_file = $Definition.source_file
        source_sha256 = $Definition.source_sha256
        model_sha256 = $weightsHash
    }
}
