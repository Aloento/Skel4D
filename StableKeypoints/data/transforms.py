"""
Data transformation utilities
"""

import math
import torch
import torch.nn.functional as F


class RandomAffineWithInverse:
    """Random affine transformation with inverse capability"""
    
    def __init__(
            self,
            degrees=0,
            scale=(1.0, 1.0),
            translate=(0.0, 0.0),
            fill_value=1.0,
            use_border_padding=False,
    ):
        self.degrees = degrees
        self.scale = scale
        self.translate = translate
        self.fill_value = fill_value
        # Optional fallback: if images already have uniform background, border padding suffices
        self.use_border_padding = use_border_padding

        # Initialize self.last_params to 0s
        self.last_params = {
            "theta": torch.eye(2, 3).unsqueeze(0),
        }

    def create_affine_matrix(self, angle, scale, translations_percent):
        angle_rad = math.radians(angle)

        # Create affine matrix
        theta = torch.tensor(
            [
                [math.cos(angle_rad), math.sin(angle_rad), translations_percent[0]],
                [-math.sin(angle_rad), math.cos(angle_rad), translations_percent[1]],
            ],
            dtype=torch.float,
        )

        theta[:, :2] = theta[:, :2] * scale
        theta = theta.unsqueeze(0)  # Add batch dimension
        return theta

    def __call__(self, img_tensor, theta=None):

        if theta is None:
            theta = []
            for i in range(img_tensor.shape[0]):
                # Calculate random parameters
                angle = torch.rand(1).item() * (2 * self.degrees) - self.degrees
                scale_factor = torch.rand(1).item() * (self.scale[1] - self.scale[0]) + self.scale[0]
                translations_percent = (
                    torch.rand(1).item() * (2 * self.translate[0]) - self.translate[0],
                    torch.rand(1).item() * (2 * self.translate[1]) - self.translate[1],
                )

                # Create the affine matrix
                theta.append(self.create_affine_matrix(
                    angle, scale_factor, translations_percent
                ))
            theta = torch.cat(theta, dim=0).to(img_tensor.device)

        # Store them for inverse transformation
        self.last_params = {
            "theta": theta,
        }

        # Apply transformation
        grid = F.affine_grid(theta, img_tensor.size(), align_corners=False).to(img_tensor.device)
        # padding_mode 'zeros' keeps performance; we'll overwrite OOB with fill_value unless border padding requested
        padding_mode = 'border' if self.use_border_padding else 'zeros'
        transformed_img = F.grid_sample(img_tensor, grid, align_corners=False, padding_mode=padding_mode)

        if not self.use_border_padding:
            # Determine out-of-bounds mask (any coord component outside [-1,1])
            oob = (grid[..., 0].abs() > 1) | (grid[..., 1].abs() > 1)
            if oob.any():
                # Broadcast fill value
                fill = self.fill_value
                if not torch.is_tensor(fill):
                    fill = torch.as_tensor(fill, dtype=transformed_img.dtype, device=transformed_img.device)
                # Shape to (B,1,1,1) then broadcast
                transformed_img = torch.where(oob.unsqueeze(1), fill.view(1, 1, 1, 1), transformed_img)

        return transformed_img

    def inverse(self, img_tensor):
        # Retrieve stored parameters
        theta = self.last_params["theta"]

        # Augment the affine matrix to make it 3x3 on the same device/dtype
        batch_size = theta.shape[0]
        aug_row = torch.zeros((batch_size, 1, 3), dtype=theta.dtype, device=theta.device)
        aug_row[:, :, 2] = 1
        theta_augmented = torch.cat([theta, aug_row], dim=1)

        # Compute the inverse of the affine matrix and take 2x3 part
        theta_inv_augmented = torch.inverse(theta_augmented)
        theta_inv = theta_inv_augmented[:, :2, :]

        # Apply inverse transformation to match img_tensor
        theta_inv = theta_inv.to(device=img_tensor.device, dtype=img_tensor.dtype)
        grid_inv = F.affine_grid(theta_inv, img_tensor.size(), align_corners=False).to(img_tensor.device)
        padding_mode = 'border' if self.use_border_padding else 'zeros'
        untransformed_img = F.grid_sample(img_tensor, grid_inv, align_corners=False, padding_mode=padding_mode)

        if not self.use_border_padding:
            oob = (grid_inv[..., 0].abs() > 1) | (grid_inv[..., 1].abs() > 1)
            if oob.any():
                fill = self.fill_value
                if not torch.is_tensor(fill):
                    fill = torch.as_tensor(fill, dtype=untransformed_img.dtype, device=untransformed_img.device)
                untransformed_img = torch.where(oob.unsqueeze(1), fill.view(1, 1, 1, 1), untransformed_img)

        return untransformed_img


def return_theta(scale, pixel_loc, rotation_angle_degrees=0):
    """
    Create affine transformation matrix
    
    Args:
        scale: Scaling factor
        pixel_loc: Pixel location between 0 and 1
        rotation_angle_degrees: Rotation angle between 0 and 360
    """

    rescaled_loc = pixel_loc * 2 - 1

    rotation_angle_radians = math.radians(rotation_angle_degrees)
    cos_theta = math.cos(rotation_angle_radians)
    sin_theta = math.sin(rotation_angle_radians)

    # Determine device and dtype from inputs when possible
    if isinstance(pixel_loc, torch.Tensor):
        device = pixel_loc.device
        dtype = pixel_loc.dtype
        res_y = rescaled_loc[1].item() if isinstance(rescaled_loc[1], torch.Tensor) else float(rescaled_loc[1])
        res_x = rescaled_loc[0].item() if isinstance(rescaled_loc[0], torch.Tensor) else float(rescaled_loc[0])
    else:
        device = None
        dtype = torch.float32
        res_y = rescaled_loc[1]
        res_x = rescaled_loc[0]

    val_scale = scale.item() if isinstance(scale, torch.Tensor) else float(scale)

    theta = torch.tensor(
        [
            [val_scale * cos_theta, -val_scale * sin_theta, res_y],
            [val_scale * sin_theta, val_scale * cos_theta, res_x],
        ],
        dtype=dtype,
        device=device,
    ).unsqueeze(0)
    return theta
