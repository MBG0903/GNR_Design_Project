"""Native-resolution draft generation and deterministic export validation.

No function in this module establishes semantic or loom production readiness.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
from PIL import Image, ImageOps

RULES = [
    'R1: Motifs interpreted correctly; pencil noise distinguished from shapes.',
    'R2: Distinct semantic pieces separated by white gaps; continuous forms preserved.',
    'R3: Intended solid pieces have no accidental white holes.',
    'R4: Intentional channels, cut-outs, eyes, veins and hollow ornaments preserved.',
    'R5: Exact red/white palette confirmed by automated validation.',
    'R6: Edges are crisp, with no hooks, blobs or random thickness changes.',
    'R7: Genuine tapers step through adjacent thickness levels without jumps.',
    'R8: Thin taper levels have longer runs than thicker levels.',
    'R9: No stray pixels or bumps outside intended curve envelopes.',
    'R10: Stems narrow gradually and flow continuously into legitimate branches.',
    'R11: Taper rules applied only to genuinely tapering structures.',
    'R12: Equivalent repeated pieces use consistent construction.',
    'R13: Motifs, composition and meaningful white spaces preserved.',
    'R14: Exact source dimensions confirmed by automated validation.',
    'R15: Topology reviewed semantically, not by component counts.',
    'R16: All uncertain regions reviewed and resolved by the designer.',
]


def rgb(image):
    """Composite transparency on white, retaining the original pixel geometry."""
    im = image if isinstance(image, Image.Image) else Image.fromarray(np.asarray(image).astype('uint8'))
    if im.mode in ('RGBA', 'LA') or 'transparency' in im.info:
        layer = im.convert('RGBA')
        bg = Image.new('RGBA', layer.size, 'white')
        return Image.alpha_composite(bg, layer).convert('RGB')
    return im.convert('RGB')


def exact_palette(image):
    """Project editor colours onto red/white by nearest RGB distance; no resizing."""
    a = np.asarray(rgb(image), dtype=np.int32)
    red = np.array([255, 0, 0], dtype=np.int32)
    white = np.array([255, 255, 255], dtype=np.int32)
    mask = ((a - red) ** 2).sum(2) < ((a - white) ** 2).sum(2)
    return render(mask)


def render(mask):
    a = np.full((*mask.shape, 3), 255, dtype=np.uint8)
    a[mask] = [255, 0, 0]
    return Image.fromarray(a)


def validate(image, source_size):
    im = rgb(image)
    a = np.asarray(im)
    red = np.all(a == [255, 0, 0], axis=2)
    white = np.all(a == [255, 255, 255], axis=2)
    palette_ok = bool(np.all(red | white))
    size_ok = tuple(im.size) == tuple(source_size)
    nonempty = bool(red.any() and white.any())
    return dict(width=im.width, height=im.height, size_ok=size_ok,
                palette_ok=palette_ok, has_red_and_white=nonempty,
                invalid_colour_pixels=int((~(red | white)).sum()),
                red_fraction=float(red.mean()),
                automated_checks_passed=bool(size_ok and palette_ok and nonempty),
                semantic_validation='Requires designer review; not inferred from component counts.')


def export_reviewed(image, source, checked_rules, reviewer, unresolved, directory):
    report = validate(image, rgb(source).size)
    if not report['automated_checks_passed']:
        raise ValueError('Export blocked: dimensions, palette, or empty-design check failed.')
    if set(checked_rules) != set(RULES) or not reviewer.strip() or unresolved.strip():
        raise ValueError('Export blocked: check all 16 rules, name the reviewer, and resolve/clear the uncertainty list.')
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    im = rgb(image)
    png, bmp, record = out/'design.png', out/'design.bmp', out/'review.json'
    im.save(png)
    im.save(bmp)
    report.update(status='designer_reviewed_raster', reviewer=reviewer.strip(),
                  reviewed_rules=list(RULES), source_rgb_sha256=hashlib.sha256(rgb(source).tobytes()).hexdigest(),
                  output_png_sha256=hashlib.sha256(png.read_bytes()).hexdigest(),
                  limitation='Raster approval only. Loom format, weave mapping, repeat alignment and machine constraints require separate confirmation.')
    record.write_text(json.dumps(report, indent=2))
    return [str(png), str(bmp), str(record)]
