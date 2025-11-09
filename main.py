import argparse
import csv
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim
from torchvision import transforms

from classes import Encoder, Generator


def get_device(force_cpu: bool = False) -> torch.device:
    if not force_cpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t - 0.5) / 0.5),  # map [0,1] -> [-1,1]
        ]
    )


def inverse_to_pil(t: torch.Tensor) -> Image.Image:
    # t expected in [-1,1]; map back to [0,1]
    t = (t.clamp(-1, 1) + 1.0) / 2.0
    return transforms.ToPILImage()(t)


def list_images(path: Path) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
    if path.is_file():
        return [path]
    return sorted([p for p in path.rglob("*") if p.suffix.lower() in exts])


def load_models(netE: Encoder, netG: Generator, channels: int, models_dir: Path, device: torch.device) -> None:
    e_path = models_dir / f"netE{channels}.model"
    g_path = models_dir / f"netG{channels}.model"
    if not e_path.exists() or not g_path.exists():
        raise FileNotFoundError(f"Missing model files: {e_path} and/or {g_path}")
    netE.load_state_dict(torch.load(str(e_path), map_location=device))
    netG.load_state_dict(torch.load(str(g_path), map_location=device))
    netE.to(device)
    netG.to(device)
    print(f"Models loaded with {channels} channels from '{models_dir}'.")


def run_inference_on_image(
    image_path: Path,
    netE: Encoder,
    netG: Generator,
    device: torch.device,
    transform: transforms.Compose,
) -> Tuple[Image.Image, Image.Image]:
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)  # [1,3,128,128], [-1,1]
    with torch.no_grad():
        compressed = netE(img_tensor)
        decompressed = netG(compressed).squeeze(0)  # [3,128,128], [-1,1]
    recon_img = inverse_to_pil(decompressed.cpu())
    return img, recon_img


def calculate_metrics(original: Image.Image, reconstructed: Image.Image) -> Tuple[float, float]:
    # Convert to numpy arrays in [0,1]
    x = np.asarray(original.resize((128, 128))).astype(np.float32) / 255.0
    y = np.asarray(reconstructed).astype(np.float32) / 255.0
    # For SSIM, if 3 channels, set channel_axis
    ssim_val = compute_ssim(x, y, data_range=1.0, channel_axis=2)
    psnr_val = compute_psnr(x, y, data_range=1.0)
    return ssim_val, psnr_val


def main() -> None:
    parser = argparse.ArgumentParser(description="Image compression/decompression with evaluation.")
    parser.add_argument("--input", type=str, required=True, help="Path to image file or directory.")
    parser.add_argument("--channels", type=int, default=8, help="Encoder channels to match model filenames (default: 8).")
    parser.add_argument("--models-dir", type=str, default="models", help="Directory with netE*.model and netG*.model.")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Where to save reconstructions and metrics.csv.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available.")
    args = parser.parse_args()

    input_path = Path(args.input)
    models_dir = Path(args.models_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device(force_cpu=args.cpu)
    transform = get_transform()

    # Initialize and load models
    netE = Encoder()
    netG = Generator()
    load_models(netE, netG, args.channels, models_dir, device)
    netE.eval()
    netG.eval()

    images = list_images(input_path)
    if not images:
        raise FileNotFoundError(f"No images found at {input_path}")

    metrics_rows = [("filename", "ssim", "psnr_db", "output_path")]
    for img_path in images:
        orig_img, recon_img = run_inference_on_image(img_path, netE, netG, device, transform)
        ssim_val, psnr_val = calculate_metrics(orig_img, recon_img)

        out_name = img_path.stem + "_recon.png"
        out_path = output_dir / out_name
        recon_img.save(out_path)

        print(f"{img_path.name}: SSIM={ssim_val:.4f}, PSNR={psnr_val:.2f} dB -> {out_path}")
        metrics_rows.append((img_path.name, f"{ssim_val:.6f}", f"{psnr_val:.4f}", str(out_path)))

    with open(output_dir / "metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(metrics_rows)
    print(f"Saved metrics to {output_dir / 'metrics.csv'}")


if __name__ == "__main__":
    main()
