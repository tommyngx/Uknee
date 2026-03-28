import io
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image


DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[1]


def _resolve_device(device: str):
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _normalize_output(output):
    if isinstance(output, (list, tuple)):
        output = output[-1]
    elif isinstance(output, dict):
        output = output["out"] if "out" in output else next(iter(output.values()))
    return output


def _to_pil_image(image_input):
    if isinstance(image_input, Image.Image):
        return image_input
    if isinstance(image_input, (str, Path)):
        return Image.open(image_input)
    if isinstance(image_input, bytes):
        return Image.open(io.BytesIO(image_input))
    if isinstance(image_input, np.ndarray):
        if image_input.ndim == 2:
            return Image.fromarray(image_input.astype(np.uint8), mode="L")
        if image_input.ndim == 3:
            return Image.fromarray(image_input.astype(np.uint8))
        raise ValueError(f"Unsupported numpy image shape: {image_input.shape}")
    raise TypeError(f"Unsupported image input type: {type(image_input)}")


def load_model(weight_path, repo_root=None, device="auto", threshold=0.5):
    if repo_root is None:
        repo_root = DEFAULT_REPO_ROOT
    repo_root = str(Path(repo_root).resolve())
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from models import build_model

    device = _resolve_device(device)
    weight_path = Path(weight_path).expanduser().resolve()

    ckpt = torch.load(weight_path, map_location=device, weights_only=False)
    cfg_dict = ckpt.get("config")
    if cfg_dict is None:
        raise ValueError("Checkpoint .pth does not contain config.")

    cfg = SimpleNamespace(**cfg_dict)
    if "RWKV" in str(cfg.model) and device.type != "cuda":
        raise RuntimeError("RWKV models in this repo require CUDA/GPU for inference.")

    model = build_model(
        cfg,
        input_channel=int(cfg.input_channel),
        num_classes=int(cfg.num_classes),
    ).to(device)

    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    return {
        "model": model,
        "config": cfg,
        "device": device,
        "threshold": float(threshold),
        "weight_path": str(weight_path),
    }


def predict_mask(runtime, image_input, threshold=None, return_pil=True, resize_back=True):
    model = runtime["model"]
    cfg = runtime["config"]
    device = runtime["device"]
    threshold = runtime["threshold"] if threshold is None else float(threshold)

    image = _to_pil_image(image_input)
    original_size = image.size

    if int(cfg.input_channel) == 1:
        resized = image.convert("L").resize((int(cfg.img_size), int(cfg.img_size)), Image.BILINEAR)
        image_array = np.array(resized, dtype=np.float32)[..., None] / 255.0
    else:
        resized = image.convert("RGB").resize((int(cfg.img_size), int(cfg.img_size)), Image.BILINEAR)
        image_array = np.array(resized, dtype=np.float32) / 255.0

    input_tensor = torch.from_numpy(image_array.transpose(2, 0, 1)).unsqueeze(0).to(device)

    with torch.no_grad():
        output = _normalize_output(model(input_tensor))
        if output.ndim != 4:
            raise ValueError(f"Unexpected output shape: {tuple(output.shape)}")

        if output.shape[1] == 1:
            probabilities = torch.sigmoid(output)[0, 0].detach().cpu().numpy()
            mask = (probabilities >= threshold).astype(np.uint8) * 255
        else:
            prediction = torch.argmax(output, dim=1)[0].detach().cpu().numpy().astype(np.uint8)
            max_label = max(int(output.shape[1] - 1), 1)
            mask = (prediction * (255 // max_label)).astype(np.uint8)

    mask_image = Image.fromarray(mask)
    if resize_back:
        mask_image = mask_image.resize(original_size, Image.NEAREST)

    if return_pil:
        return mask_image
    return np.array(mask_image, dtype=np.uint8)


def predict_to_file(runtime, image_input, output_path, threshold=None, resize_back=True):
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mask_image = predict_mask(
        runtime,
        image_input=image_input,
        threshold=threshold,
        return_pil=True,
        resize_back=resize_back,
    )
    mask_image.save(output_path)
    return str(output_path)


def predict_mask_png_bytes(runtime, image_input, threshold=None, resize_back=True):
    mask_image = predict_mask(
        runtime,
        image_input=image_input,
        threshold=threshold,
        return_pil=True,
        resize_back=resize_back,
    )
    buffer = io.BytesIO()
    mask_image.save(buffer, format="PNG")
    return buffer.getvalue()
