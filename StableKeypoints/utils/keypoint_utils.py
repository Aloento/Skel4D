"""
Keypoint detection and processing utilities
"""

import torch


def gaussian_circle(pos, size=64, sigma=16, device="cuda"):
    """Create a batch of 2D Gaussian circles with a given size, standard deviation, and center coordinates.

    pos is in between 0 and 1 and has shape [batch_size, 2]
    """
    batch_size = pos.shape[0]
    _pos = pos * size  # Shape [batch_size, 2]
    _pos = _pos.unsqueeze(1).unsqueeze(1)  # Shape [batch_size, 1, 1, 2]

    grid = torch.meshgrid(torch.arange(size).to(device), torch.arange(size).to(device))
    grid = torch.stack(grid, dim=-1) + 0.5  # Shape [size, size, 2]
    grid = grid.unsqueeze(0)  # Shape [1, size, size, 2]

    dist_sq = (grid[..., 1] - _pos[..., 1]) ** 2 + (
        grid[..., 0] - _pos[..., 0]
    ) ** 2  # Shape [batch_size, size, size]
    dist_sq = -1 * dist_sq / (2.0 * sigma**2.0)
    gaussian = torch.exp(dist_sq)  # Shape [batch_size, size, size]

    return gaussian


def gaussian_circles(pos, size=64, sigma=16, device="cuda"):
    """In the case of multiple points, pos has shape [batch_size, num_points, 2]"""

    circles = []

    for i in range(pos.shape[0]):
        _circles = gaussian_circle(
            pos[i], size=size, sigma=sigma, device=device
        )  # Assuming H and W are the same

        circles.append(_circles)

    circles = torch.stack(circles)
    circles = torch.mean(circles, dim=0)

    return circles


def find_max_pixel(map):
    """
    finds the pixel of the map with the highest value
    map shape [batch_size, h, w]

    output shape [batch_size, 2]
    """

    batch_size, h, w = map.shape

    map_reshaped = map.view(batch_size, -1)

    max_indices = torch.argmax(map_reshaped, dim=-1)

    max_indices = max_indices.view(batch_size, 1)

    max_indices = torch.cat([max_indices // w, max_indices % w], dim=-1)

    # offset by a half a pixel to get the center of the pixel
    max_indices = max_indices + 0.5

    return max_indices


def mask_radius(map, max_coords, radius):
    """
    Masks all values within a given radius of the max_coords in the map.

    Args:
    map (Tensor): The attention map with shape [batch_size, h, w].
    max_coords (Tensor): The coordinates of the point to mask around, shape [batch_size, 2].
    radius (float): The radius within which to mask the values.

    Returns:
    Tensor: The masked map.
    """
    batch_size, h, w = map.shape

    # Create a meshgrid to compute the distance for each pixel
    x_coords = torch.arange(w).view(1, -1).repeat(h, 1).to(map.device)
    y_coords = torch.arange(h).view(-1, 1).repeat(1, w).to(map.device)
    x_coords = x_coords.unsqueeze(0).repeat(batch_size, 1, 1)
    y_coords = y_coords.unsqueeze(0).repeat(batch_size, 1, 1)

    # Calculate squared Euclidean distance from the max_coords
    squared_dist = (x_coords - max_coords[:, 1].unsqueeze(1).unsqueeze(2))**2 + \
                   (y_coords - max_coords[:, 0].unsqueeze(1).unsqueeze(2))**2

    # Mask out pixels within the specified radius
    mask = squared_dist > radius**2
    masked_map = map * mask.float()

    return masked_map


def find_k_max_pixels(map, num=3):
    """
    finds the pixel of the map with the highest value
    map shape [batch_size, h, w]

    output shape [num, batch_size, 2]
    """

    batch_size, h, w = map.shape

    points = []

    for i in range(num):
        point = find_max_pixel(map)
        points.append(point)
        map = mask_radius(map, point, 0.05*h)

    return torch.stack(points)


def find_top_k_gaussian(attention_maps, top_k, sigma=3, epsilon=1e-5, num_subjects=1):
    """
    attention_maps is of shape [batch_size, image_h, image_w]

    min_dist set to 0 becomes a simple top_k
    """

    device = attention_maps.device

    batch_size, image_h, image_w = attention_maps.shape

    max_pixel_locations = find_k_max_pixels(attention_maps, num=num_subjects)/image_h

    # Normalize the activation maps to represent probability distributions
    attention_maps_softmax = torch.softmax(attention_maps.view(batch_size, image_h * image_w)+epsilon, dim=-1)

    target = gaussian_circles(max_pixel_locations, size=image_h, sigma=sigma, device=attention_maps.device)

    target = target.reshape(batch_size, image_h * image_w)+epsilon
    target/=target.sum(dim=-1, keepdim=True)

    # sort the kl distances between attention_maps_softmax and target
    kl_distances = torch.sum(target * (torch.log(target) - torch.log(attention_maps_softmax)), dim=-1)
    # get the argsort for kl_distances
    kl_distances_argsort = torch.argsort(kl_distances, dim=-1, descending=False)

    return torch.tensor(kl_distances_argsort[:top_k]).to(device)


def furthest_point_sampling(attention_maps, top_k, top_initial_candidates):
    """
    attention_maps is of shape [batch_size, image_h, image_w]

    min_dist set to 0 becomes a simple top_k
    """

    device = attention_maps.device

    batch_size, image_h, image_w = attention_maps.shape

    # Assuming you have a function find_max_pixel to get the pixel locations
    max_pixel_locations = find_max_pixel(attention_maps)/image_h

    # Find the furthest two points from top_initial_candidates
    max_dist = -1

    for i in range(len(top_initial_candidates)):
        for j in range(i+1, len(top_initial_candidates)):
            dist = torch.sqrt(torch.sum((max_pixel_locations[top_initial_candidates[i]] - max_pixel_locations[top_initial_candidates[j]])**2))
            if dist > max_dist:
                max_dist = dist
                furthest_pair = (top_initial_candidates[i].item(), top_initial_candidates[j].item())

    # Initialize the furthest point sampling with the furthest pair
    selected_indices = [furthest_pair[0], furthest_pair[1]]

    for _ in range(top_k - 2):
        max_min_dist = -1
        furthest_point = None

        for i in top_initial_candidates:
            if i.item() in selected_indices:
                continue

            this_min_dist = torch.min(torch.sqrt(torch.sum((max_pixel_locations[i] - torch.index_select(max_pixel_locations, 0, torch.tensor(selected_indices).to(device)))**2, dim=-1)))

            if this_min_dist > max_min_dist:
                max_min_dist = this_min_dist
                furthest_point = i.item()

        if furthest_point is not None:
            selected_indices.append(furthest_point)

    return torch.tensor(selected_indices).to(device)
