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
   cd img_comp_gans
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Usage
To compress and decompress an image, run the following command:
```bash
python main.py <path_to_image>
```
Replace `<path_to_image>` with the path to the image you want to process.

### Models
Ensure the pre-trained models are saved in the `models/` directory:
- `netE<channels>.model`: Encoder
- `netG<channels>.model`: Generator

## Results
- **Compression Factor**: Achieves an 84% reduction in storage requirements.
- **SSIM**: Structural Similarity Index (SSIM) values above 0.95, ensuring high-quality reconstructions.

## Conclusion
This GAN-CNN Encoder-Decoder framework offers a robust solution for high-quality image compression, addressing the limitations of conventional approaches. By combining structural and adversarial losses, the model ensures detailed and visually appealing reconstructions suitable for practical applications.
