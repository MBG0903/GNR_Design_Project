"""Reference-guided GenAI pilot; embeddings select examples, never paint pixels.

Query targets are evaluation-only. Inferred design groups do not establish an
independent benchmark until crops/variants and pair alignment are reviewed.
"""
import csv
import hashlib
import json
import re
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .data import EXTENSIONS, pair_key
from .core import validate
from .comparison import compare
from .genai import generate, POLICY

EMBEDDING_MODEL = 'clip-ViT-B-32'


def image(path):
    with Image.open(path) as opened:
        return opened.convert('RGB')


def pixel_hash(path):
    im = image(path)
    return hashlib.sha256(im.tobytes() + str(im.size).encode()).hexdigest()


def suggest_groups(stage2, destination):
    """Initial grouping by design number, retained for designer correction."""
    destination = Path(destination)
    if destination.exists():
        return destination
    keys = sorted({pair_key(p, 2) for p in Path(stage2).rglob('*')
                   if p.is_file() and p.suffix.lower() in EXTENSIONS})
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('x', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['key', 'design_id'])
        for key in keys:
            match = re.search(r'\bgnr\s*(\d+)|^(\d+)\s*gnr\b', key)
            number = next((v for v in match.groups() if v), None) if match else None
            writer.writerow([key, f'GNR_{number}' if number else key])
    return destination


def embed_sources(rows, cache, encoder=None, model_name=EMBEDDING_MODEL, progress=print):
    """Local pretrained CLIP on sketches ONLY. Cached by pixels and model name.

    Optional encoder is for deterministic tests. It must encode PIL images.
    No API image generation is involved in similarity indexing.
    """
    cache = Path(cache)
    signature = {'model': model_name, 'sources': [(r['key'], r['source_hash']) for r in rows]}
    # JSON tuples become lists, so canonicalize before comparing cached metadata.
    signature = json.loads(json.dumps(signature))
    for row in rows:
        if pixel_hash(row['source']) != row['source_hash']:
            raise ValueError('Sketch changed since audit; re-audit before indexing.')
    if cache.exists():
        with np.load(cache, allow_pickle=False) as data:
            if json.loads(str(data['signature'])) != signature:
                raise ValueError('Embedding cache belongs to another dataset/model; use a new cache file.')
            vectors = data['vectors']
    else:
        if encoder is None:
            from sentence_transformers import SentenceTransformer
            encoder = SentenceTransformer(model_name)
        batches = []
        for start in range(0, len(rows), 32):
            if progress:
                progress(f'Indexing sketches {start+1}–{min(start+32, len(rows))}/{len(rows)}')
            imgs = [image(r['source']) for r in rows[start:start+32]]
            batches.append(np.asarray(encoder.encode(imgs, batch_size=16, convert_to_numpy=True,
                                                     show_progress_bar=False), dtype=np.float32))
        if not batches:
            raise ValueError('No audited sketches to index.')
        vectors = np.concatenate(batches)
        cache.parent.mkdir(parents=True, exist_ok=True)
        with cache.open('xb') as handle:
            np.savez_compressed(handle, vectors=vectors, signature=json.dumps(signature))
    if vectors.ndim != 2 or len(vectors) != len(rows) or not np.isfinite(vectors).all():
        raise ValueError('Invalid sketch embeddings.')
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if (norms == 0).any():
        raise ValueError('Invalid sketch embeddings.')
    return vectors / norms


def plan_pilot(rows, vectors, count=5, references=2, groups_reviewed=False):
    if count < 1 or references not in (1, 2, 3):
        raise ValueError('Choose a positive query count and 1–3 reference pairs.')
    if len(vectors) != len(rows):
        raise ValueError('Embedding count differs from audited pair count.')
    # Every group and exact duplicate must belong to only one split.
    seen = {}
    for row in rows:
        for field in ('group', 'source_hash', 'target_hash'):
            token = (field, row[field])
            if token in seen and seen[token] != row['split']:
                raise ValueError('Dataset leakage: group or duplicate crosses splits.')
            seen[token] = row['split']
    queries = []
    selected_groups = set()
    for i, row in enumerate(rows):
        if row['split'] == 'validation' and row['group'] not in selected_groups:
            queries.append(i)
            selected_groups.add(row['group'])
    # Fixed seed sampling, unrelated to output quality; no selection of best drafts.
    np.random.default_rng(42).shuffle(queries)
    jobs = []
    for qi in queries[:count]:
        query = rows[qi]
        candidates = [i for i, r in enumerate(rows) if r['split'] == 'reference'
                      and r['group'] != query['group']]
        candidates.sort(key=lambda i: (-float(vectors[qi] @ vectors[i]), rows[i]['key']))
        chosen = []
        families = set()
        for i in candidates:
            if rows[i]['group'] in families:
                continue
            chosen.append({'key': rows[i]['key'], 'cosine_similarity': float(vectors[qi] @ vectors[i])})
            families.add(rows[i]['group'])
            if len(chosen) == references:
                break
        if len(chosen) < references:
            raise ValueError('Not enough independent reference families for requested examples.')
        jobs.append({'query': query['key'], 'references': chosen})
    if not jobs:
        raise ValueError('No validation designs available.')
    return {'version': 1, 'groups_reviewed': bool(groups_reviewed), 'jobs': jobs,
            'evaluation_scope': 'validation pilot; final test split is unused',
            'grouping_status': 'designer-reviewed' if groups_reviewed else 'provisional filename groups; related designs may still leak',
            'embedding_model': EMBEDDING_MODEL, 'training_performed': False}


def preview_plan(rows, plan, destination):
    lookup = {r['key']: r for r in rows}
    nrefs = max(len(j['references']) for j in plan['jobs'])
    panel = Image.new('RGB', (260*(1+2*nrefs), 230*len(plan['jobs'])), 'white')
    draw = ImageDraw.Draw(panel)
    for y, job in enumerate(plan['jobs']):
        query = lookup[job['query']]
        frames = [(f"Query: {query['key']}", query['source'])]
        for selected in job['references']:
            ref = lookup[selected['key']]
            frames += [(f"Example: {ref['key']}", ref['source']), ('Approved example output', ref['target'])]
        for x, (label, path) in enumerate(frames):
            im = image(path)
            im.thumbnail((250, 195))
            panel.paste(im, (260*x, 230*y+30))
            draw.text((260*x+3, 230*y+5), label, fill='black')
    panel.save(destination)


def _write_reports(output, summaries, plan):
    fields = ['query', 'method', 'pixel_agreement_pct', 'red_iou_pct', 'red_dice_pct']
    with (output/'scores.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)
    complete = []
    for job in plan['jobs']:
        pair = {r['method']: r for r in summaries if r['query'] == job['query']}
        if len(pair) == 2:
            complete.append(pair)
    aggregate = {'completed_design_pairs': len(complete), 'planned_designs': len(plan['jobs']),
                 'grouping_status': plan['grouping_status'], 'production_ready': False,
                 'semantic_rules_verified': False, 'test_split_used': False,
                 'note': 'One generation per method/design. Small validation pilot, not final test accuracy or rule compliance.'}
    if complete:
        aggregate['methods'] = {}
        for method in ('rules_only', 'similar_examples'):
            aggregate['methods'][method] = {
                metric: float(np.mean([pair[method][metric] for pair in complete]))
                for metric in fields[2:]}
        aggregate['delta_percentage_points'] = {
            metric: aggregate['methods']['similar_examples'][metric] - aggregate['methods']['rules_only'][metric]
            for metric in fields[2:]}
        aggregate['example_guided_designs_at_least_70pct_red_iou'] = sum(
            pair['similar_examples']['red_iou_pct'] >= 70 for pair in complete)
        aggregate['example_guided_designs_at_least_80pct_red_iou'] = sum(
            pair['similar_examples']['red_iou_pct'] >= 80 for pair in complete)
    (output/'summary.json').write_text(json.dumps(aggregate, indent=2))
    return aggregate


def run_pilot(rows, plan, output, client, model='gpt-image-2.5-sunburst', progress=print):
    """Paired rules-only / retrieved-example generation. No target-driven editing."""
    if client is None:
        raise ValueError('An authenticated image API client is required.')
    lookup = {r['key']: r for r in rows}
    if len(lookup) != len(rows):
        raise ValueError('Duplicate pair keys.')
    # Recheck leakage on the saved plan, not just at retrieval time.
    query_keys = {j['query'] for j in plan['jobs']}
    if not query_keys or len(query_keys) != len(plan['jobs']):
        raise ValueError('Missing or duplicate queries.')
    used = set(query_keys)
    for job in plan['jobs']:
        query = lookup[job['query']]
        if query['split'] != 'validation':
            raise ValueError('Pilot queries must come from validation, never final test.')
        if not 1 <= len(job['references']) <= 3:
            raise ValueError('Expected 1–3 independent examples.')
        ref_groups = set()
        for ref in job['references']:
            row = lookup[ref['key']]
            if row['split'] != 'reference' or row['group'] == query['group'] or row['group'] in ref_groups:
                raise ValueError('Reference family leakage or duplication.')
            ref_groups.add(row['group'])
            used.add(ref['key'])
    # All selected references must be distinct from ALL evaluation queries/targets.
    query_hashes = {lookup[k][field] for k in query_keys for field in ('source_hash', 'target_hash')}
    query_groups = {lookup[k]['group'] for k in query_keys}
    for key in used:
        row = lookup[key]
        for field, hashfield in [('source', 'source_hash'), ('target', 'target_hash')]:
            if pixel_hash(row[field]) != row[hashfield]:
                raise ValueError(f'{key}: input changed since audit; no API calls made.')
        if not validate(image(row['target']), image(row['source']).size)['automated_checks_passed']:
            raise ValueError(f'{key}: invalid target palette/dimensions.')
        if key not in query_keys and (row['group'] in query_groups or any(row[f] in query_hashes for f in ('source_hash', 'target_hash'))):
            raise ValueError('Held-out query/target appears in a generation reference.')
    config = {'plan': plan, 'model': model,
              'inputs': [{k: lookup[key][k] for k in ('key', 'source_hash', 'target_hash', 'group', 'split')} for key in sorted(used)],
              'policy_hash': hashlib.sha256(POLICY.encode()).hexdigest(),
              'generation_implementation_hash': hashlib.sha256(Path(generate.__code__.co_filename).read_bytes()).hexdigest(),
              'request': {'size': '1024x1024', 'quality': 'high'}}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    configpath = output/'experiment.json'
    if configpath.exists() and json.loads(configpath.read_text()) != config:
        raise ValueError('Experiment changed; use a new run folder to preserve comparisons.')
    configpath.write_text(json.dumps(config, indent=2))
    summaries = []
    for index, job in enumerate(plan['jobs']):
        query = lookup[job['query']]
        folder = output/f'design_{index+1:03d}'
        folder.mkdir(exist_ok=True)
        sketch, target = image(query['source']), image(query['target'])
        sketch.save(folder/'step_2.png')
        target.save(folder/'designer_step_3.png')
        (folder/'selection.json').write_text(json.dumps(job, indent=2))
        candidates = []
        for method in ('rules_only', 'similar_examples'):
            work = folder/method
            draft = work/'draft.png'
            completion = work/'completed.json'
            if draft.exists():
                if not completion.exists() or not (work/'generation.json').exists():
                    raise ValueError(f'{work}: incomplete request; inspect saved draft before continuing.')
                if json.loads(completion.read_text())['draft_hash'] != pixel_hash(draft):
                    raise ValueError(f'{work}: completed draft changed; preserve original benchmark output.')
            if not draft.exists():
                if work.exists():
                    raise ValueError(f'{work}: incomplete request; inspect before starting a new run.')
                refs = [] if method == 'rules_only' else [
                    (lookup[r['key']]['source'], lookup[r['key']]['target']) for r in job['references']]
                progress(f"{index+1}/{len(plan['jobs'])}: {query['key']} — {method}")
                # Persist started state before calling: prevents accidental repeat charges
                # if Colab disconnects after the provider accepted the request.
                work.mkdir()
                (work/'request_started.json').write_text(json.dumps({'status': 'started', 'method': method}))
                try:
                    generate(query['source'], work, client=client, model=model, references=refs)
                    completion.write_text(json.dumps({'draft_hash': pixel_hash(draft)}))
                except Exception as exc:
                    (work/'failure.json').write_text(json.dumps({'error_type': type(exc).__name__, 'http_status': getattr(exc, 'status_code', None)}))
                    raise RuntimeError(f'{method} request failed ({type(exc).__name__}). Completed outputs retained; inspect API usage and incomplete folder before retrying.') from None
            candidate = image(draft)
            if not validate(candidate, sketch.size)['automated_checks_passed']:
                raise ValueError('Generated raster dimensions/palette failed.')
            report, diff = compare(target, candidate)
            report.update(designer_target_sent_to_generation=False, production_ready=False,
                          semantic_rules_verified=False, grouping_status=plan['grouping_status'])
            candidate.save(work/'step_3.bmp')
            diff.save(work/'difference.png')
            (work/'comparison.json').write_text(json.dumps(report, indent=2))
            summaries.append({'query': query['key'], 'method': method,
                              **{f'{metric}_pct': 100*report[metric] for metric in ('pixel_agreement', 'red_iou', 'red_dice')}})
            candidates.append(candidate)
            _write_reports(output, summaries, plan)
        frames = [('Sketch', sketch), ('Designer', target), ('Rules only', candidates[0]), ('Similar examples', candidates[1])]
        w, h = sketch.size
        panel = Image.new('RGB', (w*4, h+30), 'white')
        draw = ImageDraw.Draw(panel)
        for x, (label, im) in enumerate(frames):
            draw.text((x*w+3, 5), label, fill='black')
            panel.paste(im, (x*w, 30))
        panel.save(folder/'comparison.png')
    summary = _write_reports(output, summaries, plan)
    with zipfile.ZipFile(output/'pilot_review.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output.rglob('*')):
            if path.is_file() and path.suffix != '.zip':
                archive.write(path, path.relative_to(output))
    return summary
