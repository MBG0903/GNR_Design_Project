"""Strict pairing/audit and reproducible whole-design splits. No silent resizing."""
from pathlib import Path
import csv
import hashlib
import json
import random
import re
from PIL import Image
from .core import rgb, validate

EXTENSIONS = {'.bmp', '.png', '.tif', '.tiff', '.jpg', '.jpeg'}

def pair_key(path, stage):
    stem = Path(path).stem.casefold().strip()
    # Also support 'GNR 3001 ( FILE 2 )' and variant suffixes after FILE.
    stem = re.sub(rf'\(\s*file\s*{stage}\s*\)', ' ', stem)
    # Example: '362 GNR 2' and '362 GNR 3' -> '362 gnr'.
    stem = re.sub(rf'[ _-]+(?:(?:stage|step)[ _-]*)?{stage}$', '', stem)
    return re.sub(r'[ _-]+', ' ', stem).strip()

def audit_dataset(stage2, stage3, output, seed=42, groups_csv=None, require_groups=False, hold_fraction=0.15):
    if not 0 < hold_fraction < 0.5:
        raise ValueError('hold_fraction must be between 0 and 0.5.')
    roots = [Path(stage2), Path(stage3)]
    for root in roots:
        if not root.is_dir():
            raise ValueError(f'Dataset directory not found: {root}')
    indexes, errors = [], []
    for root, stage in zip(roots, [2,3]):
        index = {}
        for p in sorted(root.rglob('*')):
            if p.is_file() and p.suffix.lower() in EXTENSIONS:
                key = pair_key(p, stage)
                if not key or key in index:
                    errors.append(f'Ambiguous filename key: {key!r} in {root}')
                else:
                    index[key] = p
        indexes.append(index)
    a,b = indexes
    for key in sorted(set(a)^set(b)):
        errors.append(f'Unmatched: {key} (missing Stage_{3 if key in a else 2})')
    grouping = {}
    if groups_csv and Path(groups_csv).is_file():
        with Path(groups_csv).open(newline='') as handle:
            for entry in csv.DictReader(handle):
                key, group = entry.get('key', '').strip(), entry.get('design_id', '').strip()
                if key in grouping:
                    errors.append(f'Duplicate design-group mapping: {key}')
                grouping[key] = group
    if require_groups:
        missing = [key for key in sorted(set(a)&set(b)) if not grouping.get(key)]
        if missing:
            output_path = Path(output)
            output_path.mkdir(parents=True, exist_ok=True)
            template = output_path/'groups_template.csv'
            with template.open('w', newline='') as handle:
                writer = csv.writer(handle)
                writer.writerow(['key', 'design_id'])
                writer.writerows((key, grouping.get(key,'')) for key in sorted(set(a)&set(b)))
            errors.append(f'{len(missing)} pairs need design_id values. Fill {template}, save as groups.csv and rerun.')
    rows = []
    for key in sorted(set(a)&set(b)):
        try:
            with Image.open(a[key]) as im:
                source = rgb(im)
            with Image.open(b[key]) as im:
                target = rgb(im)
            checks = validate(target, source.size)
            if not checks['automated_checks_passed']:
                errors.append(f'{key}: invalid target or dimensions: {checks}')
            rows.append(dict(key=key, source=str(a[key].resolve()), target=str(b[key].resolve()),
                             width=source.width, height=source.height,
                             source_hash=hashlib.sha256(source.tobytes()+str(source.size).encode()).hexdigest(),
                             target_hash=hashlib.sha256(target.tobytes()+str(target.size).encode()).hexdigest()))
        except Exception as exc:
            errors.append(f'{key}: cannot read pair: {exc}')
    # Put exact duplicate inputs OR outputs in the same split, transitively.
    parent = list(range(len(rows)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    seen = {}
    for i,row in enumerate(rows):
        for field in ['source_hash', 'target_hash', 'design_id']:
            value = grouping.get(row['key']) if field == 'design_id' else row[field]
            if not value:
                continue
            token = (field,value)
            if token in seen:
                parent[find(i)] = find(seen[token])
            seen[token] = i
    groups = {}
    for i in range(len(rows)):
        groups.setdefault(find(i), []).append(i)
    group_list = list(groups.values())
    random.Random(seed).shuffle(group_list)
    if len(group_list) < 3:
        errors.append('Need at least three independent design groups for reference/validation/test.')
    hold = min(max(1, round(len(group_list)*hold_fraction)), max(1,(len(group_list)-1)//2))
    for j,group in enumerate(group_list):
        split = 'test' if j < hold else 'validation' if j < 2*hold else 'reference'
        for i in group:
            rows[i]['split'] = split
            rows[i]['group'] = str(find(i))
    report = dict(seed=seed, hold_fraction=hold_fraction, pairs=len(rows), independent_groups=len(group_list), errors=errors,
                  counts={s:sum(r.get('split')==s for r in rows) for s in ['reference','validation','test']},
                  note='Whole-design mapping and exact duplicate inputs/outputs grouped transitively. Inspect contact sheets to verify correspondence.')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output/'audit.json').write_text(json.dumps(report,indent=2))
    if errors:
        # Never leave a stale successful manifest available after a failed re-audit.
        (output/'manifest.json').unlink(missing_ok=True)
        raise ValueError(f'Audit failed with {len(errors)} issue(s); inspect {output / "audit.json"}')
    (output/'manifest.json').write_text(json.dumps(rows,indent=2))
    return report

def contact_sheet(rows, destination, count=12):
    """Preview pairs side by side; thumbnails are for inspection only, never training."""
    from PIL import ImageDraw
    selected = rows[:count]
    canvas = Image.new('RGB',(512,max(1,len(selected))*220),'#eeeeee')
    draw = ImageDraw.Draw(canvas)
    for i,row in enumerate(selected):
        draw.text((4,i*220+2),row['key'],fill='black')
        for j,field in enumerate(['source','target']):
            with Image.open(row[field]) as opened:
                im = rgb(opened)
            im.thumbnail((250,195))
            canvas.paste(im,(j*256,i*220+23))
    canvas.save(destination)
    return str(destination)
