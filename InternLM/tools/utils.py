import numpy as np
import torch
from PIL import Image
from torchvision import transforms

encode_transform = transforms.Compose(
    [
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
    ]
)


def convert_decode_to_pil(rec_image):
    rec_image = 2.0 * rec_image - 1.0
    rec_image = torch.clamp(rec_image, -1.0, 1.0)
    rec_image = (rec_image + 1.0) / 2.0
    rec_image *= 255.0
    rec_image = rec_image.permute(0, 2, 3, 1).detach().cpu().numpy().astype(np.uint8)
    pil_images = [Image.fromarray(image) for image in rec_image]
    return pil_images


def patchify(imgs, p):
    """
    imgs: (N, C, H, W)
    x: (N, L, patch_size**2 * C)
    """
    assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

    in_chans = imgs.shape[1]
    h = w = imgs.shape[2] // p
    x = imgs.reshape(shape=(imgs.shape[0], in_chans, h, p, w, p))
    x = torch.einsum('nchpwq->nhwpqc', x)
    x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * in_chans))
    return x


def unpatchify(x, p):
    """
    x: (N, L, patch_size**2 * C)
    imgs: (N, C, H, W)
    """
    # p = self.patch_embed.patch_size[0]
    h = w = int(x.shape[1] ** .5)
    assert h * w == x.shape[1]

    x = x.reshape(shape=(x.shape[0], h, w, p, p, -1))
    x = torch.einsum('nhwpqc->nchpwq', x)
    imgs = x.reshape(shape=(x.shape[0], -1, h * p, h * p))
    return imgs