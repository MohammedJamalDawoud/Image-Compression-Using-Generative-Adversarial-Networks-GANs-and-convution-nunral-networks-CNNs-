# Image Compression Using GAN-CNN Encoder-Decoder Framework

## Architecture Overview
1. **Encoder**: Utilizes convolutional layers with residual connections inspired by ResNet. This design ensures efficient feature extraction while maintaining memory efficiency.
2. **Generator (Decoder)**: Focuses on reconstructing the image from the compressed latent representation, with layers designed for up-sampling and detailed texture synthesis.
3. **Discriminator**: A key GAN component that takes real and reconstructed image pairs as input. It distinguishes between them and provides adversarial feedback, enhancing the perceptual quality of the reconstructed images.

## How to Use
### Prerequisites
- Python 3.9
- PyTorch
- torchvision
- PIL (Pillow)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/MohammedJamalDawoud/Image-Compression-Using-Generative-Adversarial-Networks-GANs-and-convution-nunral-networks-CNNs-.git
   cd Image-Compression-Using-Generative-Adversarial-Networks-GANs-and-convution-nunral-networks-CNNs-
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Usage
Compress and decompress image(s) with evaluation:
```bash
python main.py --input path/to/image_or_folder --channels 8 --models-dir models --output-dir outputs
```

Options:
- `--input`: Path to an image file or a directory of images.
- `--channels`: Encoder channel count that matches the model filenames (default: 8).
- `--models-dir`: Directory containing `netE<channels>.model` and `netG<channels>.model` (default: `models`).
- `--output-dir`: Where to write reconstructed images and a CSV of metrics (default: `outputs/`).
- `--cpu`: Force CPU even if CUDA is available.

Notes:
- Inputs are normalized to [-1, 1] to match training; outputs are mapped back to [0, 1] before saving.
- The script computes PSNR and SSIM per image and writes a summary CSV.

### Models
Ensure the pre-trained models are saved in the `models/` directory:
- `netE<channels>.model`: Encoder
- `netG<channels>.model`: Generator

## Results
- Example metrics (with provided demo weights):
  - **SSIM**: Structural Similarity Index above 0.95 on held-out samples
  - **PSNR**: Typically >30 dB on 128×128 reconstructions

## Conclusion
This GAN-CNN Encoder-Decoder framework offers a robust solution for high-quality image compression, addressing the limitations of conventional approaches. By combining structural and adversarial losses, the model ensures detailed and visually appealing reconstructions suitable for practical applications.
