from __future__ import annotations

import hashlib
import json
import math
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


RUST_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = RUST_ROOT.parent
sys.path.insert(0, str(REPOSITORY_ROOT))

from data_utils.feather_hubert.feather_hubert import (  # noqa: E402
    FeatherHuBERTConfig,
    FeatherHuBERTEncoder,
)
from unet import Model as ProductionUnet  # noqa: E402


ARCHIVE_NAME = "burn-feasibility-v1.zip"
ARCHIVE_LIMIT_BYTES = 20 * 1024 * 1024
FIXED_ZIP_TIME = (2026, 8, 17, 0, 0, 0)
SEED = 20260817
MIN_FORWARD_SENSITIVITY = 1e-3
MIN_GRADIENT_OR_UPDATE = 1e-6
MIN_L1_RESIDUAL_MARGIN = 1e-3
TRAIN_INPUT_SEED = 3
REQUIRED_PYTHON = (3, 10, 16)
REQUIRED_TORCH = "2.1.2+cu118"
REQUIRED_NUMPY = "1.23.5"


def deterministic_values(
    name: str,
    tensor: torch.Tensor,
    normalization_weights: set[str],
    normalization_biases: set[str],
) -> torch.Tensor:
    if not tensor.is_floating_point():
        return torch.zeros_like(tensor)
    if name.endswith("running_var") or name in normalization_weights:
        return torch.ones_like(tensor)
    if (
        name.endswith("running_mean")
        or name in normalization_biases
        or name.endswith("bias")
    ):
        return torch.zeros_like(tensor)

    fan_in = math.prod(tensor.shape[1:]) if tensor.ndim >= 2 else tensor.numel()
    seed = int.from_bytes(
        hashlib.sha256(name.encode("utf-8")).digest()[:8], "little"
    ) & ((1 << 63) - 1)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    codes = torch.randint(0, 8, tensor.shape, generator=generator)
    codebook = torch.tensor(
        [-7.0, -5.0, -3.0, -1.0, 1.0, 3.0, 5.0, 7.0],
        dtype=torch.float32,
    ) / math.sqrt(21.0)

    if name.startswith("fuse_conv."):
        gain = 1.25
    elif name.startswith("audio_model."):
        gain = 1.0
    else:
        gain = 0.5
    scale = gain * math.sqrt(2.0 / max(fan_in, 1))
    return (codebook[codes] * scale).to(dtype=tensor.dtype)


def fill_state_dict(module: torch.nn.Module) -> None:
    normalization_weights: set[str] = set()
    normalization_biases: set[str] = set()
    for module_name, child in module.named_modules():
        if isinstance(child, (nn.modules.batchnorm._BatchNorm, nn.GroupNorm)):
            prefix = f"{module_name}." if module_name else ""
            if child.weight is not None:
                normalization_weights.add(f"{prefix}weight")
            if child.bias is not None:
                normalization_biases.add(f"{prefix}bias")

    state = module.state_dict()
    for name, tensor in state.items():
        tensor.copy_(
            deterministic_values(
                name,
                tensor,
                normalization_weights,
                normalization_biases,
            )
        )
    module.load_state_dict(state)


def max_abs_difference(left: torch.Tensor, right: torch.Tensor) -> float:
    return torch.max(torch.abs(left - right)).item()


def require_metric(name: str, value: float, minimum: float) -> None:
    if not math.isfinite(value) or value < minimum:
        raise RuntimeError(f"{name} is too small: {value} < {minimum}")


class InvertedResidual(nn.Module):
    def __init__(
        self,
        inp: int,
        oup: int,
        stride: int,
        use_res_connect: bool,
        expand_ratio: int = 6,
    ) -> None:
        super().__init__()
        if stride not in (1, 2):
            raise ValueError(f"unsupported stride: {stride}")
        hidden_dim = inp * expand_ratio
        self.use_res_connect = use_res_connect
        self.conv = nn.Sequential(
            nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                3,
                stride,
                1,
                groups=hidden_dim,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        output = self.conv(value)
        return value + output if self.use_res_connect else output


class DoubleConvDw(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 2) -> None:
        super().__init__()
        self.double_conv = nn.Sequential(
            InvertedResidual(
                in_channels,
                out_channels,
                stride=stride,
                use_res_connect=False,
                expand_ratio=2,
            ),
            InvertedResidual(
                out_channels,
                out_channels,
                stride=1,
                use_res_connect=True,
                expand_ratio=2,
            ),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.double_conv(value)


class InConvDw(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.inconv = InvertedResidual(
            in_channels,
            out_channels,
            stride=1,
            use_res_connect=False,
            expand_ratio=2,
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.inconv(value)


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.maxpool_conv = DoubleConvDw(in_channels, out_channels, stride=2)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(value)


class Up(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConvDw(in_channels, out_channels, stride=1)

    def forward(self, value: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        value = self.up(value)
        diff_y = skip.shape[2] - value.shape[2]
        diff_x = skip.shape[3] - value.shape[3]
        value = F.pad(
            value,
            [
                diff_x // 2,
                diff_x - diff_x // 2,
                diff_y // 2,
                diff_y - diff_y // 2,
            ],
        )
        return self.conv(torch.cat([value, skip], dim=1))


class OutConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.conv(value)


class AudioConvHubert(nn.Module):
    def __init__(self, channels: list[int]) -> None:
        super().__init__()
        self.conv1 = InvertedResidual(
            16, channels[1], stride=1, use_res_connect=False, expand_ratio=2
        )
        self.conv2 = InvertedResidual(
            channels[1],
            channels[2],
            stride=1,
            use_res_connect=False,
            expand_ratio=2,
        )
        self.conv3 = nn.Conv2d(
            channels[2], channels[3], kernel_size=3, padding=1, stride=(2, 2)
        )
        self.bn3 = nn.BatchNorm2d(channels[3])
        self.conv4 = InvertedResidual(
            channels[3],
            channels[3],
            stride=1,
            use_res_connect=True,
            expand_ratio=2,
        )
        self.conv5 = nn.Conv2d(
            channels[3], channels[4], kernel_size=3, padding=3, stride=2
        )
        self.bn5 = nn.BatchNorm2d(channels[4])
        self.relu = nn.ReLU()
        self.conv6 = InvertedResidual(
            channels[4],
            channels[4],
            stride=1,
            use_res_connect=True,
            expand_ratio=2,
        )
        self.conv7 = InvertedResidual(
            channels[4],
            channels[4],
            stride=1,
            use_res_connect=True,
            expand_ratio=2,
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = self.conv1(value)
        value = self.conv2(value)
        value = self.relu(self.bn3(self.conv3(value)))
        value = self.conv4(value)
        value = self.relu(self.bn5(self.conv5(value)))
        value = self.conv6(value)
        return self.conv7(value)


class ConfigurableUnet(nn.Module):
    def __init__(self, channels: list[int]) -> None:
        super().__init__()
        self.audio_model = AudioConvHubert(channels)
        self.fuse_conv = nn.Sequential(
            DoubleConvDw(channels[4] * 2, channels[4], stride=1),
            DoubleConvDw(channels[4], channels[3], stride=1),
        )
        self.inc = InConvDw(6, channels[0])
        self.down1 = Down(channels[0], channels[1])
        self.down2 = Down(channels[1], channels[2])
        self.down3 = Down(channels[2], channels[3])
        self.down4 = Down(channels[3], channels[4])
        self.up1 = Up(channels[4], channels[3] // 2)
        self.up2 = Up(channels[3], channels[2] // 2)
        self.up3 = Up(channels[2], channels[1] // 2)
        self.up4 = Up(channels[1], channels[0])
        self.outc = OutConv(channels[0], 3)

    def forward(self, image: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        image1 = self.inc(image)
        image2 = self.down1(image1)
        image3 = self.down2(image2)
        image4 = self.down3(image3)
        image5 = self.down4(image4)
        audio = self.audio_model(audio)
        image5 = self.fuse_conv(torch.cat([image5, audio], dim=1))
        output = self.up1(image5, image4)
        output = self.up2(output, image3)
        output = self.up3(output, image2)
        output = self.up4(output, image1)
        return torch.sigmoid(self.outc(output))


def repeating_pattern(
    shape: tuple[int, ...], modulus: int, offset: int, divisor: float
) -> torch.Tensor:
    values = torch.arange(np.prod(shape), dtype=torch.float32)
    values = (values.remainder(modulus) - offset) / divisor
    return values.reshape(shape)


def make_l1_safe_target(
    prediction: torch.Tensor, target: torch.Tensor, margin: float
) -> tuple[torch.Tensor, int]:
    """Move only cusp-near target elements while preserving their signs."""
    residual = prediction - target
    cusp = residual.abs() <= margin
    direction = torch.where(
        residual >= 0.0,
        torch.ones_like(residual),
        -torch.ones_like(residual),
    )
    distance = margin * 2.0
    preferred = prediction - direction * distance
    fallback = prediction + direction * distance
    safe = torch.where(
        (preferred >= 0.0) & (preferred <= 1.0),
        preferred,
        fallback,
    )
    adjusted = torch.where(cusp, safe, target)
    if adjusted.min().item() < 0.0 or adjusted.max().item() > 1.0:
        raise RuntimeError("L1-safe target left the normalized [0, 1] range")
    return adjusted, int(cusp.sum().item())


def save_array(path: Path, tensor: torch.Tensor) -> dict[str, object]:
    array = tensor.detach().cpu().numpy().astype(np.float32, copy=False)
    np.save(path, array, allow_pickle=False)
    return {"path": path.relative_to(path.parents[1]).as_posix(), "shape": list(array.shape)}


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def save_checkpoint(
    path: Path, model: nn.Module, config: dict[str, object], epoch: int = 0
) -> None:
    torch.save(
        {"epoch": epoch, "model": model.state_dict(), "config": config},
        path,
    )


def generate_tiny_checkpoints(weights: Path) -> None:
    tiny_state = {
        "weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
        "bias": torch.tensor([0.25, -0.5], dtype=torch.float32),
    }
    torch.save(tiny_state, weights / "tiny_direct.pth")
    torch.save({"model": tiny_state}, weights / "tiny_nested.pth")
    torch.save(
        {"model": {"weight": tiny_state["weight"]}},
        weights / "tiny_missing.pth",
    )
    torch.save(
        {"model": {**tiny_state, "unexpected": torch.ones(1)}},
        weights / "tiny_unexpected.pth",
    )


def generate_feather_fixture(work: Path) -> dict[str, object]:
    config = FeatherHuBERTConfig(
        channels=32,
        expansion=2,
        num_blocks=2,
        output_dim=64,
        dropout=0.0,
    )
    model = FeatherHuBERTEncoder(config).cpu().eval()
    fill_state_dict(model)
    waveform = torch.linspace(-0.75, 0.75, 1360, dtype=torch.float32).reshape(
        1, 1360
    )
    with torch.no_grad():
        output = model(waveform)
        zero_output = model(torch.zeros_like(waveform))
    sensitivity = max_abs_difference(output, zero_output)
    require_metric("waveform_vs_zero_max_abs", sensitivity, MIN_FORWARD_SENSITIVITY)

    save_checkpoint(
        work / "weights/feather_micro.pth",
        model,
        {
            "channels": 32,
            "expansion": 2,
            "num_blocks": 2,
            "output_dim": 64,
            "dropout": 0.0,
        },
    )
    save_array(work / "arrays/feather_input.npy", waveform)
    save_array(work / "arrays/feather_output.npy", output)
    print(f"feather_micro_eval input={tuple(waveform.shape)} output={tuple(output.shape)}")
    return {
        "kind": "feather_hubert",
        "weights": "weights/feather_micro.pth",
        "config": {
            "channels": 32,
            "expansion": 2,
            "num_blocks": 2,
            "output_dim": 64,
            "dropout": 0.0,
        },
        "inputs": {"waveform": "arrays/feather_input.npy"},
        "expected": {"output": "arrays/feather_output.npy"},
        "metrics": {"waveform_vs_zero_max_abs": sensitivity},
    }


def generate_production_unet_fixture(work: Path) -> dict[str, object]:
    channels = [32, 64, 128, 256, 512]
    model = ProductionUnet(n_channels=6, mode="hubert").cpu().eval()
    fill_state_dict(model)
    image = repeating_pattern((1, 6, 160, 160), 251, 0, 250.0)
    audio = repeating_pattern((1, 16, 32, 32), 127, 63, 64.0)
    with torch.no_grad():
        output = model(image, audio)
        alternate_image = model(1.0 - image, audio)
        alternate_audio = model(image, -audio)
    image_sensitivity = max_abs_difference(output, alternate_image)
    audio_sensitivity = max_abs_difference(output, alternate_audio)
    print(
        "unet_production_sensitivity "
        f"image={image_sensitivity:.9f} audio={audio_sensitivity:.9f}"
    )
    require_metric("image_branch_max_abs", image_sensitivity, MIN_FORWARD_SENSITIVITY)
    require_metric("audio_branch_max_abs", audio_sensitivity, MIN_FORWARD_SENSITIVITY)

    save_checkpoint(
        work / "weights/unet_production.pth",
        model,
        {"channels": channels, "mode": "hubert", "n_channels": 6},
    )
    save_array(work / "arrays/unet_image.npy", image)
    save_array(work / "arrays/unet_audio.npy", audio)
    save_array(work / "arrays/unet_output.npy", output)
    print(
        "unet_production_eval "
        f"image={tuple(image.shape)} audio={tuple(audio.shape)} output={tuple(output.shape)}"
    )
    return {
        "kind": "original_unet",
        "weights": "weights/unet_production.pth",
        "config": {"channels": channels, "mode": "hubert", "n_channels": 6},
        "inputs": {
            "image": "arrays/unet_image.npy",
            "audio": "arrays/unet_audio.npy",
        },
        "expected": {"output": "arrays/unet_output.npy"},
        "metrics": {
            "image_branch_max_abs": image_sensitivity,
            "audio_branch_max_abs": audio_sensitivity,
        },
    }


def save_named_state_arrays(
    work: Path, state: dict[str, torch.Tensor], names: Iterable[str], prefix: str
) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for index, name in enumerate(names):
        path = work / f"arrays/{prefix}_{index:02d}.npy"
        metadata = save_array(path, state[name])
        result[name] = metadata
    return result


def generate_micro_train_fixture(work: Path) -> dict[str, object]:
    channels = [2, 4, 8, 16, 32]
    model = ConfigurableUnet(channels).cpu()
    fill_state_dict(model)

    generator = torch.Generator(device="cpu").manual_seed(TRAIN_INPUT_SEED)
    image = torch.rand(
        (1, 6, 160, 160), dtype=torch.float32, device="cpu", generator=generator
    )
    audio = (
        torch.rand(
            (1, 16, 32, 32), dtype=torch.float32, device="cpu", generator=generator
        )
        * 2.0
        - 1.0
    )
    target = repeating_pattern((1, 3, 160, 160), 89, 0, 88.0)
    preview_model = ConfigurableUnet(channels).cpu()
    preview_model.load_state_dict(model.state_dict())
    preview_model.train()
    with torch.no_grad():
        preview = preview_model(image, audio)
    target, l1_cusp_adjusted_elements = make_l1_safe_target(
        preview, target, MIN_L1_RESIDUAL_MARGIN
    )

    save_checkpoint(
        work / "weights/unet_micro_train.pth",
        model,
        {"channels": channels, "mode": "hubert", "n_channels": 6},
    )
    save_array(work / "arrays/train_image.npy", image)
    save_array(work / "arrays/train_audio.npy", audio)
    save_array(work / "arrays/train_target.npy", target)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
    )
    model.train()
    optimizer.zero_grad(set_to_none=True)
    parameter_names = [
        "inc.inconv.conv.0.weight",
        "audio_model.conv1.conv.0.weight",
        "outc.conv.weight",
    ]
    named_parameters = dict(model.named_parameters())
    before_step = {
        name: named_parameters[name].detach().clone() for name in parameter_names
    }
    prediction = model(image, audio)
    initial_residual_min_abs = torch.min(torch.abs(prediction - target)).item()
    require_metric(
        "initial_l1_residual_min_abs",
        initial_residual_min_abs,
        MIN_L1_RESIDUAL_MARGIN,
    )
    initial_loss_tensor = torch.mean(torch.abs(prediction - target))
    initial_loss_tensor.backward()
    gradient_max_abs = {
        name: named_parameters[name].grad.detach().abs().max().item()
        for name in parameter_names
    }
    optimizer.step()
    update_max_abs = {
        name: (named_parameters[name].detach() - before_step[name]).abs().max().item()
        for name in parameter_names
    }
    for name in parameter_names:
        require_metric(
            f"gradient_max_abs[{name}]",
            gradient_max_abs[name],
            MIN_GRADIENT_OR_UPDATE,
        )
        require_metric(
            f"update_max_abs[{name}]",
            update_max_abs[name],
            MIN_GRADIENT_OR_UPDATE,
        )

    model.eval()
    with torch.no_grad():
        post_step_loss = torch.mean(torch.abs(model(image, audio) - target)).item()

    state = model.state_dict()
    batch_norm_names = [
        "inc.inconv.conv.1.running_mean",
        "inc.inconv.conv.1.running_var",
        "audio_model.conv1.conv.1.running_mean",
        "audio_model.conv1.conv.1.running_var",
    ]
    parameters = save_named_state_arrays(work, state, parameter_names, "train_parameter")
    batch_norm = save_named_state_arrays(work, state, batch_norm_names, "train_batch_norm")
    expected = {
        "initial_loss": initial_loss_tensor.detach().item(),
        "post_step_loss": post_step_loss,
        "post_step_mode": "eval",
        "parameters": parameters,
        "batch_norm_state": batch_norm,
    }
    write_json(work / "arrays/train_expected.json", expected)
    metrics = {
        "image_parameter_gradient_max_abs": gradient_max_abs[parameter_names[0]],
        "audio_parameter_gradient_max_abs": gradient_max_abs[parameter_names[1]],
        "output_parameter_gradient_max_abs": gradient_max_abs[parameter_names[2]],
        "image_parameter_update_max_abs": update_max_abs[parameter_names[0]],
        "audio_parameter_update_max_abs": update_max_abs[parameter_names[1]],
        "output_parameter_update_max_abs": update_max_abs[parameter_names[2]],
        "initial_l1_residual_min_abs": initial_residual_min_abs,
        "l1_cusp_adjusted_elements": float(l1_cusp_adjusted_elements),
    }
    print(
        "unet_micro_train_step "
        f"image={tuple(image.shape)} audio={tuple(audio.shape)} target={tuple(target.shape)} "
        f"initial_loss={expected['initial_loss']:.9f} post_step_loss={post_step_loss:.9f}"
    )
    return {
        "kind": "original_unet_train_step",
        "weights": "weights/unet_micro_train.pth",
        "config": {"channels": channels, "mode": "hubert", "n_channels": 6},
        "optimizer": {
            "type": "adam",
            "learning_rate": 1e-3,
            "beta1": 0.9,
            "beta2": 0.999,
            "epsilon": 1e-8,
            "weight_decay": 0.0,
        },
        "loss": "mean_absolute_error",
        "inputs": {
            "image": "arrays/train_image.npy",
            "audio": "arrays/train_audio.npy",
            "target": "arrays/train_target.npy",
        },
        "expected_json": "arrays/train_expected.json",
        "metrics": metrics,
    }


def write_deterministic_archive(work: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.unlink(missing_ok=True)
    with zipfile.ZipFile(
        output_path,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
        allowZip64=True,
    ) as archive:
        for path in sorted(item for item in work.rglob("*") if item.is_file()):
            relative = path.relative_to(work).as_posix()
            info = zipfile.ZipInfo(relative, date_time=FIXED_ZIP_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED)


def main() -> None:
    actual_python = sys.version_info[:3]
    if actual_python != REQUIRED_PYTHON:
        raise RuntimeError(f"expected Python {REQUIRED_PYTHON}, got {actual_python}")
    if torch.__version__ != REQUIRED_TORCH:
        raise RuntimeError(f"expected torch {REQUIRED_TORCH}, got {torch.__version__}")
    if np.__version__ != REQUIRED_NUMPY:
        raise RuntimeError(f"expected numpy {REQUIRED_NUMPY}, got {np.__version__}")

    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    np.random.seed(SEED)

    output_dir = RUST_ROOT / "tests/golden"
    output_path = output_dir / ARCHIVE_NAME
    sidecar_path = output_path.with_suffix(".sha256")

    with tempfile.TemporaryDirectory(prefix="feathertalk-golden-") as temp:
        work = Path(temp)
        (work / "weights").mkdir()
        (work / "arrays").mkdir()
        generate_tiny_checkpoints(work / "weights")
        fixtures = {
            "feather_micro_eval": generate_feather_fixture(work),
            "unet_production_eval": generate_production_unet_fixture(work),
            "unet_micro_train_step": generate_micro_train_fixture(work),
        }
        manifest = {
            "schema_version": 1,
            "fixture_set": "burn-feasibility-v1",
            "seed": SEED,
            "generator": {
                "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                "torch": torch.__version__,
                "numpy": np.__version__,
                "device": "cpu",
                "train_input_seed": TRAIN_INPUT_SEED,
                "train_input_dtype": "float32",
            },
            "fixtures": fixtures,
        }
        write_json(work / "manifest.json", manifest)
        write_deterministic_archive(work, output_path)

    archive_size = output_path.stat().st_size
    if archive_size > ARCHIVE_LIMIT_BYTES:
        output_path.unlink(missing_ok=True)
        raise RuntimeError(f"golden archive is too large: {archive_size} bytes")
    digest = hashlib.sha256(output_path.read_bytes()).hexdigest()
    sidecar_path.write_text(digest + "\n", encoding="ascii", newline="\n")
    print(f"archive={output_path}")
    print(f"size={archive_size}")
    print(f"sha256={digest}")


if __name__ == "__main__":
    main()
