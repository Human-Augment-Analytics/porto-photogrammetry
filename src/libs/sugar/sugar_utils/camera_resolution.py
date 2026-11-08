def camera_resolution(width, height, resolution=-1):
    """Match vanilla 3DGS input sizing, with nonzero dimensions for tiny images."""
    if width <= 0 or height <= 0:
        raise ValueError("Camera dimensions must be positive")
    if resolution != -1 and resolution <= 0:
        raise ValueError("resolution must be -1 or positive")
    if resolution in (1, 2, 4, 8):
        return max(1, round(width / resolution)), max(1, round(height / resolution))
    if resolution == -1:
        scale = width / 1600 if width > 1600 else 1
    else:
        scale = width / resolution
    return max(1, int(width / scale)), max(1, int(height / scale))