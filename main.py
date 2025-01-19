import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import argparse
from classes import *
# Define transformations for input images
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize to a fixed size
    transforms.ToTensor(),          # Convert to tensor
])

# Encoder and Generator class implementations
def compress_and_save(image_path, netE, netG):
    # Load image
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension

    # Compress using encoder
    compressed = netE(img_tensor)

    # Decompress using generator
    decompressed = netG(compressed)

    # Save decompressed image
    output_path = "decompressed.png"
    decompressed_img = transforms.ToPILImage()(decompressed.squeeze(0))
    decompressed_img.save(output_path)
    print(f"Decompressed image saved to {output_path}")


def load_models(netE, netG, num_channels_in_encoder):
    netE.load_state_dict(torch.load(os.path.join("models", f"netE{num_channels_in_encoder}.model")))
    netG.load_state_dict(torch.load(os.path.join("models", f"netG{num_channels_in_encoder}.model")))
    print(f"Models loaded with {num_channels_in_encoder} channels from the 'models' folder.")


def main():
    parser = argparse.ArgumentParser(description="Image compression and decompression script.")
    parser.add_argument("image_path", type=str, help="Path to the input image.")
    args = parser.parse_args()

    num_channels_in_encoder = 64  # Match the number of channels defined in the model

    # Initialize models
    netE = Encoder()  
    netG = Generator()

    # Load pre-trained models
    load_models(netE, netG, num_channels_in_encoder)

    # Set models to evaluation mode
    netE.eval()
    netG.eval()

    # Compress and decompress image
    compress_and_save(args.image_path, netE, netG)


if __name__ == "__main__":
    main()
