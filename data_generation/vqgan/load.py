from typing import List, Tuple

import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from PIL import Image
from torch import Tensor
from torchvision import transforms
from torchvision.transforms import Lambda

from .muse import VQGANModel


class ForwardWrapper(nn.Module):
    def __init__(self, vq_model, func='encode'):
        super(ForwardWrapper, self).__init__()
        self.vq_model = vq_model
        self.func = func
    
    def forward(self, x):
        return getattr(self.vq_model, self.func)(x)


def load_model(path):
    # Load the pre-trained vq model from the hub
    vq_model = VQGANModel.from_pretrained(path)
    return vq_model


def load_encoder(path):
    vq_model = load_model(path)
    encoder = ForwardWrapper(vq_model)
    return encoder


def load_decoder(path):
    vq_model = load_model(path)
    decoder = ForwardWrapper(vq_model, func='decode')
    return decoder


def load_decoder_code(path):
    vq_model = load_model(path)
    decoder = ForwardWrapper(vq_model, func='decode_code')
    return decoder


def convert_decode_to_pil(rec_image):
    rec_image = 2.0 * rec_image - 1.0
    rec_image = torch.clamp(rec_image, -1.0, 1.0)
    rec_image = (rec_image + 1.0) / 2.0
    rec_image *= 255.0
    rec_image = rec_image.permute(0, 2, 3, 1).detach().cpu().numpy().astype(np.uint8)
    pil_images = [Image.fromarray(image) for image in rec_image]
    return pil_images


class SixCrop(torch.nn.Module):
    def __init__(self, crop_size):
        super().__init__()
        self.crop_size = crop_size
    
    # def get_dimensions(self, img):
    #     """Returns the dimensions of an image as [channels, height, width].
    #
    #     Args:
    #         img (PIL Image or Tensor): The image to be checked.
    #
    #     Returns:
    #         List[int]: The image dimensions.
    #     """
    #     if isinstance(img, torch.Tensor):
    #         return F_t.get_dimensions(img)
    #
    #     return F_pil.get_dimensions(img)
    
    def get_dimensions(self, img) -> List[int]:
        if hasattr(img, "getbands"):
            channels = len(img.getbands())
        else:
            channels = img.channels
        width, height = img.size
        return [channels, height, width]
    
    def six_crop(self, img: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Crop the given image into four corners and the central crop.
        If the image is torch Tensor, it is expected
        to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions

        .. Note::
            This transform returns a tuple of images and there may be a
            mismatch in the number of inputs and targets your ``Dataset`` returns.

        Args:
            img (PIL Image or Tensor): Image to be cropped.
            size (sequence or int): Desired output size of the crop. If size is an
                int instead of sequence like (h, w), a square crop (size, size) is
                made. If provided a sequence of length 1, it will be interpreted as (size[0], size[0]).

        Returns:
           tuple: tuple (tl, tr, bl, br, center)
           Corresponding top left, top right, bottom left, bottom right and center crop.
        """
        # if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        #     _log_api_usage_once(five_crop)
        
        crop_height, crop_width = self.crop_size
        _, image_height, image_width = self.get_dimensions(img)
        
        # if crop_width > image_width or crop_height > image_height:
        # msg = "Requested crop size {} is bigger than input size {}"
        # raise ValueError(msg.format(self.crop_size, (image_height, image_width)))
        
        if crop_width > image_width:
            crop_width = image_width
            crop_height = image_width
        
        if crop_height > image_height:
            crop_width = image_height
            crop_height = image_height
        
        tl = F.crop(img, 0, 0, crop_height, crop_width)
        tr = F.crop(img, 0, image_width - crop_width, crop_height, crop_width)
        bl = F.crop(img, image_height - crop_height, 0, crop_height, crop_width)
        br = F.crop(img, image_height - crop_height, image_width - crop_width, crop_height, crop_width)
        
        if image_height > image_width:
            center_top = int(round((image_height - crop_height) / 2.0))
            cl = F.crop(img, center_top, 0, crop_height, crop_width)
            cr = F.crop(img, center_top, image_width - crop_width, crop_height, crop_width)
            return tl, tr, cl, cr, bl, br
        else:
            center_left = int(round((image_width - crop_width) / 2.0))
            ct = F.crop(img, 0, center_left, crop_height, crop_width)
            cb = F.crop(img, image_height - crop_height, center_left, crop_height, crop_width)
            return tl, tr, ct, bl, br, cb
        
        # center = center_crop(img, [crop_height, crop_width])
    
    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be scaled.

        Returns:
            PIL Image or Tensor: Rescaled image.
        """
        return self.six_crop(img)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.crop_size})"


def six_crop_encode_transform(crop_size):
    t = transforms.Compose(
            [
                    SixCrop(crop_size),
                    # transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
                    Lambda(lambda crops:
                           [transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR)(crop) for crop
                            in crops]),
                    Lambda(lambda crops: [transforms.ToTensor()(crop) for crop in crops]),
            ]
    )
    return t


encode_transform = transforms.Compose(
        [
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(256),
                transforms.ToTensor(),
        ]
)

encode_transform_no_crop = transforms.Compose(
        [
                transforms.Resize([256, 256], interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
        ]
)

encode_transform_2 = transforms.Compose(
        [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(180),
                transforms.RandomResizedCrop(256, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
        ]
)

encode_transform_rain_random = transforms.Compose(
        [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomResizedCrop(256, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
        ]
)

encode_transform_rain_random_2 = transforms.Compose(
        [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomCrop(400),
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
        ]
)

if __name__ == '__main__':
    import numpy as np
    
    vq_model = load_model('/cache/ckpt/vqgan-f16-8192-laion')
    
    image = Image.open("ILSVRC2012_val_00040846.JPEG")
    pixel_values = encode_transform(image).unsqueeze(0)
    quantized_states, indices = vq_model.encode(pixel_values)
    rec_image = vq_model.decode(quantized_states)
    pil_images = convert_decode_to_pil(rec_image)
