from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
import torch
import torch.nn as nn
from torchvision import transforms
import io
import sys

# ------------------------------------------------------------------
# Model definitions in this module
# ------------------------------------------------------------------
class Generator(nn.Module):
    def __init__(self, latent_size, image_size, hidden_size):
        super(Generator, self).__init__()
        # Generator network: latent -> feature maps -> RGB image
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_size, hidden_size * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_size * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden_size * 8, hidden_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_size * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden_size * 4, hidden_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_size * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden_size * 2, hidden_size,   4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden_size,   3,             4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)

class Discriminator(nn.Module):
    def __init__(self, image_size, hidden_size):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, hidden_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_size, hidden_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_size * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_size * 2, hidden_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_size * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_size * 4, hidden_size * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_size * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_size * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# ------------------------------------------------------------------
# Make this module __main__ so torch.load can find classes
# ------------------------------------------------------------------
_this_mod = sys.modules[__name__]
sys.modules['__main__'] = _this_mod

# ------------------------------------------------------------------
# Whitelist classes for safe unpickling
# ------------------------------------------------------------------
from torch.serialization import add_safe_globals
add_safe_globals([Generator, Discriminator])

# ------------------------------------------------------------------
# Device & hyperparameters
# ------------------------------------------------------------------
# Your checkpoint was trained with latent_size=100
device      = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
latent_size = 100  # match the checkpoint's first ConvTranspose2d in_dim
image_size  = 64
hidden_size = 64

# ------------------------------------------------------------------
# Helper to remap checkpoint keys (prefix 'main.' -> 'net.')
# ------------------------------------------------------------------
def remap_keys(state_dict, old_prefix, new_prefix):
    remapped = {}
    for k, v in state_dict.items():
        if k.startswith(old_prefix + '.'):
            remapped[new_prefix + k[len(old_prefix):]] = v
        else:
            remapped[k] = v
    return remapped

# ------------------------------------------------------------------
# Load Generator
# ------------------------------------------------------------------
generator = Generator(latent_size, image_size, hidden_size).to(device)
try:
    loaded = torch.load(
        "generator_full_final.pth",
        map_location=device,
        weights_only=False
    )
    # Determine how to extract state_dict
    if isinstance(loaded, dict) and all(isinstance(v, torch.Tensor) for v in loaded.values()):
        state = remap_keys(loaded, 'main', 'net')
    elif isinstance(loaded, nn.Module):
        state = loaded.state_dict()
        state = remap_keys(state, 'main', 'net')
    elif isinstance(loaded, dict) and 'state_dict' in loaded:
        state = loaded['state_dict']
        state = remap_keys(state, 'main', 'net')
    else:
        raise ValueError("Unrecognized format for generator checkpoint")
    generator.load_state_dict(state)
    if isinstance(generator, nn.DataParallel):
        generator = generator.module
    generator.to(device)
except Exception as e:
    raise RuntimeError(f"Error loading generator_full_final.pth: {e}")
generator.eval()

# ------------------------------------------------------------------
# Load Discriminator
# ------------------------------------------------------------------
discriminator = Discriminator(image_size, hidden_size).to(device)
try:
    loaded = torch.load(
        "discriminator_full_final.pth",
        map_location=device,
        weights_only=False
    )
    if isinstance(loaded, dict) and all(isinstance(v, torch.Tensor) for v in loaded.values()):
        state = remap_keys(loaded, 'main', 'net')
    elif isinstance(loaded, nn.Module):
        state = loaded.state_dict()
        state = remap_keys(state, 'main', 'net')
    elif isinstance(loaded, dict) and 'state_dict' in loaded:
        state = loaded['state_dict']
        state = remap_keys(state, 'main', 'net')
    else:
        raise ValueError("Unrecognized format for discriminator checkpoint")
    discriminator.load_state_dict(state)
    if isinstance(discriminator, nn.DataParallel):
        discriminator = discriminator.module
    discriminator.to(device)
except Exception as e:
    raise RuntimeError(f"Error loading discriminator_full_final.pth: {e}")
discriminator.eval()

# ------------------------------------------------------------------
# FastAPI app
# ------------------------------------------------------------------
app = FastAPI(debug=True)

@app.get("/generate/{prompt}")
async def generate_image(prompt: str):
    try:
        # Optionally seed by prompt
        noise = torch.randn(1, latent_size, 1, 1, device=device)
        with torch.no_grad():
            fake = generator(noise).cpu().squeeze(0)
        fake = (fake + 1) / 2
        fake = fake.clamp(0, 1)
        img = transforms.ToPILImage()(fake)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        return StreamingResponse(buf, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
