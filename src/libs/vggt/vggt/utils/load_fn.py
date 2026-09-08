# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from PIL import Image
from torchvision import transforms as TF
import numpy as np


def load_and_preprocess_images_square(image_path_list, mask_path_list=None, target_size=1024):
    """
    Load and preprocess images by center padding to square and resizing to target size.
    Also returns the position information of original pixels after transformation.

    Args:
        image_path_list (list): List of paths to image files
        mask_path_list (list, optional): List of paths to mask files. When provided,
            pixels outside the mask are zeroed out and the mask is transformed alongside
            the image.
        target_size (int, optional): Target size for both width and height. Defaults to 1024.

    Returns:
        tuple: (
            torch.Tensor: Batched tensor of preprocessed images with shape (N, 3, target_size, target_size),
            torch.Tensor: Array of shape (N, 6) containing [x1, y1, x2, y2, width, height] for each image,
            list: List of PIL masks (or None entries) transformed to match the output images
        )

    Raises:
        ValueError: If the input list is empty
    """
    if len(image_path_list) == 0:
        raise ValueError("At least 1 image is required")

    images = []
    masks = []
    original_coords = []
    to_tensor = TF.ToTensor()

    for i, image_path in enumerate(image_path_list):
        mask_path = None
        mask = None
        if mask_path_list is not None and i < len(mask_path_list):
            mask_path = mask_path_list[i]

        img = Image.open(image_path)

        # Black, not white: downstream confidence and keypoint logic treat 0 as "nothing here".
        if img.mode == "RGBA":
            background = Image.new("RGBA", img.size, (0, 0, 0, 255))
            img = Image.alpha_composite(background, img)

        img = img.convert("RGB")

        # The mask is carried alongside the image, never composited into it
        if mask_path is not None:
            mask = Image.open(mask_path).convert("L")

        width, height = img.size
        max_dim = max(width, height)

        left = (max_dim - width) // 2
        top = (max_dim - height) // 2

        scale = target_size / max_dim

        x1 = left * scale
        y1 = top * scale
        x2 = (left + width) * scale
        y2 = (top + height) * scale

        original_coords.append(np.array([x1, y1, x2, y2, width, height]))

        square_img = Image.new("RGB", (max_dim, max_dim), (0, 0, 0))
        square_img.paste(img, (left, top))
        square_img = square_img.resize((target_size, target_size), Image.Resampling.BICUBIC)

        if mask is not None:
            square_mask = Image.new("L", (max_dim, max_dim), 0)
            square_mask.paste(mask, (left, top))
            mask = square_mask.resize((target_size, target_size), Image.Resampling.NEAREST)

        img_tensor = to_tensor(square_img)
        images.append(img_tensor)
        masks.append(mask)

    images = torch.stack(images)
    original_coords = torch.from_numpy(np.array(original_coords)).float()

    if len(image_path_list) == 1:
        if images.dim() == 3:
            images = images.unsqueeze(0)
            original_coords = original_coords.unsqueeze(0)

    return images, original_coords, masks


def load_and_preprocess_images(image_path_list, mask_path_list=None, mode="crop"):
    """
    A quick start function to load and preprocess images for model input.
    This assumes the images should have the same shape for easier batching, but our model can also work well with different shapes.

    Args:
        image_path_list (list): List of paths to image files
        mask_path_list (list, optional): List of paths to mask files
        mode (str, optional): Preprocessing mode, either "crop" or "pad".
                             - "crop" (default): Sets width to 518px and center crops height if needed.
                             - "pad": Preserves all pixels by making the largest dimension 518px
                               and padding the smaller dimension to reach a square shape.

    Returns:
        torch.Tensor: Batched tensor of preprocessed images with shape (N, 3, H, W)

    Raises:
        ValueError: If the input list is empty or if mode is invalid

    Notes:
        - Images with different dimensions will be padded with white (value=1.0)
        - A warning is printed when images have different shapes
        - When mode="crop": The function ensures width=518px while maintaining aspect ratio
          and height is center-cropped if larger than 518px
        - When mode="pad": The function ensures the largest dimension is 518px while maintaining aspect ratio
          and the smaller dimension is padded to reach a square shape (518x518)
        - Dimensions are adjusted to be divisible by 14 for compatibility with model requirements
    """
    # Check for empty list
    if len(image_path_list) == 0:
        raise ValueError("At least 1 image is required")

    # Validate mode
    if mode not in ["crop", "pad"]:
        raise ValueError("Mode must be either 'crop' or 'pad'")

    images = []
    masks = []
    shapes = set()
    to_tensor = TF.ToTensor()
    target_size = 518

    # First process all images and collect their shapes
    for i in range(len(image_path_list)):
        image_path = image_path_list[i]
        if mask_path_list is not None and i < len(mask_path_list):
            mask_path = mask_path_list[i]
        else:
            mask_path = None
            mask = None

        # Open image
        img = Image.open(image_path)

        # Black, not white: downstream confidence and keypoint logic treat 0 as "nothing here".
        if img.mode == "RGBA":
            background = Image.new("RGBA", img.size, (0, 0, 0, 255))
            img = Image.alpha_composite(background, img)

        img = img.convert("RGB")

        # The mask is carried alongside the image, never composited into it
        if mask_path is not None:
            mask = Image.open(mask_path).convert("L")

        width, height = img.size

        if mode == "pad":
            # Make the largest dimension 518px while maintaining aspect ratio
            if width >= height:
                new_width = target_size
                new_height = round(height * (new_width / width) / 14) * 14  # Make divisible by 14
            else:
                new_height = target_size
                new_width = round(width * (new_height / height) / 14) * 14  # Make divisible by 14
        else:  # mode == "crop"
            # Original behavior: set width to 518px
            new_width = target_size
            # Calculate height maintaining aspect ratio, divisible by 14
            new_height = round(height * (new_width / width) / 14) * 14

        # Resize with new dimensions (width, height)
        img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
        if mask is not None:
            mask = mask.resize((new_width, new_height), Image.Resampling.NEAREST)
        img = to_tensor(img)  # Convert to tensor (0, 1)

        # Center crop height if it's larger than 518 (only in crop mode)
        if mode == "crop" and new_height > target_size:
            start_y = (new_height - target_size) // 2
            img = img[:, start_y : start_y + target_size, :]
            if mask is not None:
                mask = mask[:, start_y : start_y + target_size]

        # For pad mode, pad to make a square of target_size x target_size
        if mode == "pad":
            h_padding = target_size - img.shape[1]
            w_padding = target_size - img.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                # Pad with white (value=1.0)
                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )

        shapes.add((img.shape[1], img.shape[2]))
        images.append(img)
        masks.append(mask)

    # Check if we have different shapes
    # In theory our model can also work well with different shapes
    if len(shapes) > 1:
        print(f"Warning: Found images with different shapes: {shapes}")
        # Find maximum dimensions
        max_height = max(shape[0] for shape in shapes)
        max_width = max(shape[1] for shape in shapes)

        # Pad images if necessary
        padded_images = []
        for img in images:
            h_padding = max_height - img.shape[1]
            w_padding = max_width - img.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )
            padded_images.append(img)
        images = padded_images

    images = torch.stack(images)  # concatenate images

    # Ensure correct shape when single image
    if len(image_path_list) == 1:
        # Verify shape is (1, C, H, W)
        if images.dim() == 3:
            images = images.unsqueeze(0)

    return images, masks
