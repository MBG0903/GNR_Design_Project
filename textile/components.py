"""Native raster placement primitives; no resampling or inferred approval."""
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage


def rgb_image(value):
    if isinstance(value, (str, Path)):
        with Image.open(value) as im:
            return im.convert('RGB')
    return value.convert('RGB')


def binary(value):
    im = rgb_image(value)
    a = np.asarray(im)
    red = (a == [255, 0, 0]).all(axis=2)
    white = (a == [255, 255, 255]).all(axis=2)
    if not (red | white).all():
        raise ValueError('Step_3 must contain only pure red and white. Original pixels were not changed.')
    return im, red


def raster(mask):
    a = np.full((*mask.shape, 3), 255, dtype=np.uint8)
    a[mask] = [255, 0, 0]
    return Image.fromarray(a)


def digest(im):
    return hashlib.sha256(im.tobytes()+str(im.size).encode()).hexdigest()


def place_component(canvas, component, protect, x, y, existing_protection=None):
    """Red-only union, protected-white constraints, minimum 2 white pixels.

    Distance is defined on an 8-neighbour (Chebyshev) native pixel grid: red
    centres must be at least 3 steps apart. Wider intentional spaces stay wider.
    Distinct placements only; joining a continuous stem is deliberately unsupported.
    """
    base, existing = binary(canvas)
    _, red = binary(component)
    protect = np.asarray(protect, dtype=bool)
    if protect.shape != red.shape or (protect & red).any() or not red.any():
        raise ValueError('Invalid component masks.')
    if not float(x).is_integer() or not float(y).is_integer():
        raise ValueError('Placement coordinates must be integers.')
    x, y = int(x), int(y)
    h, w = red.shape
    if x < 0 or y < 0 or x+w > base.width or y+h > base.height:
        raise ValueError('The complete native component must fit; clipping/resizing is not permitted.')
    protected = np.zeros_like(existing) if existing_protection is None else np.asarray(existing_protection, dtype=bool)
    if protected.shape != existing.shape or (protected & existing).any():
        raise ValueError('Invalid canvas protected-white map.')
    added = np.zeros_like(existing); added[y:y+h, x:x+w] = red
    new_protection = np.zeros_like(existing); new_protection[y:y+h, x:x+w] = protect
    if (added & existing).any():
        raise ValueError('Placement overlaps existing red pixels.')
    if (added & protected).any() or (new_protection & existing).any():
        raise ValueError('Placement fills an intentional protected white area.')
    minimum_gap = None
    if existing.any():
        distances = ndimage.distance_transform_cdt(~existing, metric='chessboard')
        minimum_gap = int(distances[added].min())-1
        if minimum_gap < 2:
            raise ValueError(f'Placement leaves only {minimum_gap} white pixel(s); at least two are required.')
    result = raster(existing | added)
    return result, protected | new_protection, dict(x=x, y=y, native_size=[w,h],
        minimum_white_gap_to_existing=minimum_gap, spacing_metric='8-neighbour/Chebyshev: red-centre distance minus one',
        changed_pixels=int(added.sum()), transform='integer_translation_only',
        note='Minimum clearance checked. Wider spaces preserved. Motif placement and full composition need designer review.')
