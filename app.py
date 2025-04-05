from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
import torch
import numpy as np
from torchvision.utils import save_image
import torch.nn.functional as F
from io import BytesIO
from PIL import Image, ImageEnhance
from collections import OrderedDict
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import os

# Generator settings
latent_size = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the Generator class
class Generator(nn.Module):
    def __init__(self, latent_size, image_size, hidden_size):
        super(Generator, self).__init__()
        self.latent_size = latent_size
        self.image_size = image_size
        self.hidden_size = hidden_size

        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_size, hidden_size * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(hidden_size * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden_size * 8, hidden_size * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_size * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden_size * 4, hidden_size * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_size * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden_size * 2, hidden_size, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden_size, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)

# Initialize FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model (use your pre-trained model paths here)
generator = Generator(latent_size, 64, 64)
state_dict_G = torch.load('G.ckpt', map_location=device)

# Modify and load the state dict
new_state_dict_G = OrderedDict()
for k, v in state_dict_G.items():
    new_state_dict_G['net.' + k] = v

generator.load_state_dict(new_state_dict_G)
generator.eval()

# Denormalize function for image
def denorm(img_tensor):
    return img_tensor * 0.3 + 0.5

# Upscale the image using bicubic interpolation
def upscale_image_bicubic(image_tensor, scale_factor=2):
    return F.interpolate(image_tensor, scale_factor=scale_factor, mode='bicubic', align_corners=False)

# Enhance vibrancy of the image
def enhance_vibrancy(image):
    pil_img = Image.fromarray((image * 255).astype(np.uint8))
    enhancer = ImageEnhance.Color(pil_img)
    pil_img = enhancer.enhance(1.5)
    enhancer = ImageEnhance.Contrast(pil_img)
    pil_img = enhancer.enhance(1.2)
    return np.array(pil_img) / 255.0

# Endpoint for generating image based on prompt (integer input)
@app.get("/generate/{prompt}")
async def generate_image(prompt: int):
    try:
        # Generate new latent vector from user input (prompt)
        latent = torch.randn(1, latent_size, 1, 1, device=device)

        with torch.no_grad():
            fake_image = generator(latent)

        # Denorm and Upscale
        fake_image = denorm(fake_image.clamp(0, 1))
        fake_image_upscaled = upscale_image_bicubic(fake_image, scale_factor=2)

        # Prepare image for display
        image_np = fake_image_upscaled.squeeze().cpu().numpy()
        image_np = np.transpose(image_np, (1, 2, 0))

        # If specific index, apply vibrancy enhancement (you can modify conditions here)
        if prompt == 12:
            image_np = enhance_vibrancy(image_np)

        # Convert to PIL image for returning
        pil_img = Image.fromarray((image_np * 255).astype(np.uint8))

        # Save to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            pil_img.save(tmp_file, format="PNG")
            tmp_path = tmp_file.name

        # Return the image as a response
        return FileResponse(tmp_path, media_type="image/png")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Welcome to the AI GAN Image Generation API!"}
