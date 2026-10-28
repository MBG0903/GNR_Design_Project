"""Resumable image generation and explicitly unreviewed composite previews."""
import json
import zipfile
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from .tiles import read
from .tile_genai import propose, proposal_input
from .core import validate


def _latest(directory, key):
    folder = Path(directory) / 'ai_proposals' / key
    revisions = [int(p.name) for p in folder.iterdir()
                 if p.is_dir() and p.name.isdigit()] if folder.exists() else []
    return max(revisions) if revisions else None


def generate_all(directory, client, model='gpt-image-2.5-sunburst', progress=None):
    """One paid request per missing tile. Completed proposals are reused on resume.

    Stop on the first error. Earlier completed tiles remain on disk. An existing
    proposal from a different requested model is rejected rather than relabelled.
    """
    if client is None:
        raise ValueError('Connect an authenticated image API client before batch generation.')
    manifest, base = read(directory)
    for index, tile in enumerate(manifest['tiles'], 1):
        key = tile['id']; revision = _latest(directory, key)
        if revision is None:
            if progress: progress(f'{index}/{len(manifest["tiles"])}: generating {key}')
            _, metadata = propose(directory, key, base.crop(tile['context']), client, model)
        else:
            _, metadata = proposal_input(directory, key, revision, manifest)
            if metadata['requested_model'] != model:
                raise ValueError(f'{key}: existing proposal uses a different model; use a new run.')
            if progress: progress(f'{index}/{len(manifest["tiles"])}: reusing {key}')
    return assemble_preview(directory)


def assemble_preview(directory):
    """Copy all latest AI cores into a preview without inventing designer sign-off.

    This intentionally does not call the reviewed-output merge or save_proposal.
    Missing/tampered tiles block output. No preview is eligible as training truth.
    """
    directory = Path(directory); manifest, base = read(directory)
    combined = base.copy(); candidates = []; records = []
    for tile in manifest['tiles']:
        key = tile['id']; revision = _latest(directory, key)
        if revision is None:
            raise ValueError(f'{key}: missing AI proposal; complete all tiles before combining.')
        candidate, metadata = proposal_input(directory, key, revision, manifest)
        original = base.crop(tile['context'])
        checks = validate(candidate, original.size)
        if not checks['size_ok'] or not checks['palette_ok']:
            raise ValueError(f'{key}: invalid native size or palette.')
        delta = np.any(np.asarray(candidate) != np.asarray(original), axis=2)
        x0, y0, x1, y1 = tile['core_local']
        delta[y0:y1, x0:x1] = False
        if delta.any(): raise ValueError(f'{key}: proposal changed its read-only halo.')
        core = candidate.crop(tile['core_local'])
        combined.paste(core, tuple(tile['core'][:2]))
        candidates.append((key, candidate, core, metadata))
        records.append(dict(tile_id=key, revision=revision, requested_model=metadata['requested_model'],
                            candidate_rgb_sha256=metadata['candidate_rgb_sha256'],
                            core=tile['core'], context=tile['context']))
    # A distinct versioned preview never overwrites a reviewed output or source.
    folder = directory / 'ai_batch_previews'; folder.mkdir(exist_ok=True)
    versions = [int(p.name) for p in folder.iterdir() if p.is_dir() and p.name.isdigit()]
    output = folder / f'{max(versions, default=0)+1:04d}'; output.mkdir()
    (output / 'tiles').mkdir()
    combined.save(output / 'combined_unreviewed.png')
    combined.save(output / 'combined_unreviewed.bmp')
    for key, candidate, core, metadata in candidates:
        candidate.save(output / 'tiles' / f'{key}_context.png')
        core.save(output / 'tiles' / f'{key}_core.png')
        (output / 'tiles' / f'{key}.json').write_text(json.dumps(metadata, indent=2))
    comparison = Image.new('RGB', (base.width*2, base.height), 'white')
    comparison.paste(base, (0, 0)); comparison.paste(combined, (base.width, 0))
    comparison.save(output / 'before_left_after_right.png')
    guide = combined.resize((base.width*4, base.height*4), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(guide)
    for tile in manifest['tiles']:
        x0, y0, x1, y1 = tile['core']
        draw.rectangle((x0*4, y0*4, x1*4-1, y1*4-1), outline='#0066ff', width=2)
        draw.text((x0*4+4, y0*4+4), tile['id'], fill='black', stroke_width=2, stroke_fill='white')
    guide.save(output / 'tile_seam_guide_diagnostic.png')
    report = dict(status='unreviewed_ai_batch_preview', tile_count=len(records), tiles=records,
                  native_size=list(combined.size), source_sha256=manifest['draft_rgb_sha256'],
                  changed_pixels=int(np.any(np.asarray(base) != np.asarray(combined), axis=2).sum()),
                  raster_checks=validate(combined, base.size), designer_approved=False,
                  production_ready=False, semantic_hierarchy_verified=False,
                  whole_image_seam_review_required=True,
                  note='Exact native core assembly. Blue grid is diagnostic only. Left comparison is the original dense draft. No rule compliance is implied.')
    (output / 'report.json').write_text(json.dumps(report, indent=2))
    (output / 'README.txt').write_text('UNREVIEWED AI PROPOSALS\ncombined_unreviewed.png/.bmp: native design size.\n'
        'tiles/*_core.png: disjoint native pieces; *_context.png: overlapping read-only context.\n'
        'before_left_after_right.png: dense original on left, proposal composite on right.\n'
        'tile_seam_guide_diagnostic.png: enlarged inspection guide, not textile output.\n'
        'Designer review of complete curves, hierarchy, gaps, composition and seams remains required.\n')
    with zipfile.ZipFile(output / 'all_tiles_and_combined_review.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output.rglob('*')):
            if path.is_file() and path.suffix != '.zip': archive.write(path, str(path.relative_to(output)))
    return output, report
