"""Apply bounded vision-agent polygon corrections in Colab, with audit trails.

No global morphology or automatic gap cutting: those can destroy intentional
white structure or legitimate continuous forms. Semantic approval stays human.
"""
import hashlib
import json
from pathlib import Path
from PIL import Image, ImageDraw
from .core import rgb, validate
from .photoshop import validate_plan


def apply_plan(current, plan, output, source):
    output=Path(output)
    if output.exists():
        raise ValueError('Use a new output directory for each correction iteration.')
    with Image.open(current) as opened:
        before=rgb(opened)
    with Image.open(source) as original:
        source_size=original.size
    if not validate(before,source_size)['automated_checks_passed']:
        raise ValueError('Current raster must have exact red/white colours and valid content.')
    validate_plan(plan,before.size)
    after=before.copy()
    draw=ImageDraw.Draw(after)
    for op in plan['operations']:
        # Coordinates identify native pixel centres; no antialiasing/resampling.
        draw.polygon([tuple(point) for point in op['points']],
                     fill=(255,0,0) if op['colour']=='red' else (255,255,255))
    checks=validate(after,source_size)
    if not checks['automated_checks_passed']:
        raise ValueError('Corrections failed raster checks; no result saved.')
    output.mkdir(parents=True)
    after.save(output/'corrected.png');after.save(output/'corrected.bmp')
    checks.update(status='needs_semantic_review',unresolved=plan['unresolved'],
                  input_sha256=hashlib.sha256(Path(current).read_bytes()).hexdigest(),
                  output_sha256=hashlib.sha256((output/'corrected.png').read_bytes()).hexdigest(),
                  plan=plan,semantic_approval=False)
    (output/'audit.json').write_text(json.dumps(checks,indent=2))
    return str(output/'corrected.png')
