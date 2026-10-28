"""Provider-neutral correction examples; capture does not train model weights."""
import hashlib,json,shutil,tempfile,zipfile
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from PIL import Image
from .core import RULES
from .tiles import read,_load,merge

RULE_IDS={f'R{i}' for i in range(1,17)}


def validate_annotation(review,changed):
    group=review.get('design_group_id')
    if not isinstance(group,str) or not group.strip():
        raise ValueError('Enter an original design/group ID; use the same ID for all variants and crops.')
    rules=review.get('rules_corrected')
    if not isinstance(rules,list) or any(r not in RULE_IDS for r in rules) or len(set(rules))!=len(rules):
        raise ValueError('Select valid, distinct corrected rule IDs.')
    if changed and (not rules or not isinstance(review.get('correction_notes'),str) or not review['correction_notes'].strip()):
        raise ValueError('A changed example needs rule tags and a short explanation of the correction.')


def capture(directory,key,record):
    """Snapshot one approved revision with immutable before/after/sketch/masks."""
    directory=Path(directory);record=Path(record).resolve()
    folder=(directory/'proposals'/key).resolve()
    if record.parent!=folder or not record.stem.isdigit() or record.suffix!='.json':
        raise ValueError('Expected a versioned proposal review record in this run.')
    review=json.loads(record.read_text());proposal=_load(record.with_suffix('.png'))
    manifest,base=read(directory)
    tiles={t['id']:t for t in manifest['tiles']}
    if key not in tiles:raise ValueError('Unknown tile ID.')
    tile=tiles[key];initial=base.crop(tile['context']);before=initial.copy();ai_metadata=None
    ai_revision=review.get('ai_proposal_revision')
    if ai_revision is not None:
        from .tile_genai import proposal_input
        before,ai_metadata=proposal_input(directory,key,ai_revision,manifest)
        if before.size!=initial.size:raise ValueError('AI candidate dimensions do not match tile.')
    sketch=_load(directory/'sketch.png').crop(tile['context'])
    merge(directory,{key:proposal},{key:review})
    changed=np.any(np.asarray(before)!=np.asarray(proposal),axis=2)
    # Even unchanged/no-op review examples must be explicitly signed.
    required=['complete_front_back_and_tips_reviewed','hierarchy_reviewed','two_pixel_gaps_reviewed']
    if not all(review.get(k) is True for k in required) or not isinstance(review.get('reviewer'),str) or not review['reviewer'].strip() or review.get('unresolved')!=[]:
        raise ValueError('Only designer-reviewed, resolved examples may be collected.')
    validate_annotation(review,bool(changed.any()))
    core=np.zeros(changed.shape,np.uint8);x0,y0,x1,y1=tile['core_local'];core[y0:y1,x0:x1]=255
    revision=record.stem
    raw_hash=lambda im:hashlib.sha256(str(im.size).encode()+im.tobytes()).hexdigest()
    sample_id=hashlib.sha256((manifest['draft_rgb_sha256']+raw_hash(_load(directory/'sketch.png'))+review['design_group_id'].strip()+key+revision+raw_hash(before)+raw_hash(proposal)).encode()).hexdigest()[:24]
    rules_snapshot=json.dumps(RULES,ensure_ascii=False)
    data=dict(schema_version=1,sample_id=sample_id,design_group_id=review['design_group_id'].strip(),
              tile_id=key,revision=revision,source_full_size=[manifest['width'],manifest['height']],
              context_box=tile['context'],core_box=tile['core'],core_local_box=tile['core_local'],
              before_rgb_sha256=raw_hash(before),after_rgb_sha256=raw_hash(proposal),sketch_rgb_sha256=raw_hash(sketch),
              full_draft_rgb_sha256=manifest['draft_rgb_sha256'],initial_ai_draft_rgb_sha256=raw_hash(initial),
              ai_proposal_revision=ai_revision,ai_proposal_metadata=ai_metadata,
              full_sketch_rgb_sha256=raw_hash(_load(directory/'sketch.png')),
              reviewed_by=review['reviewer'].strip(),rules_corrected=review['rules_corrected'],
              correction_notes=review.get('correction_notes',''),changed_pixels=int(changed.sum()),
              example_kind='designer_correction' if changed.any() else 'designer_approved_no_change',
              draft_generator_model=ai_metadata['requested_model'] if ai_metadata else review.get('draft_generator_model') or 'unknown',
              collected_at_utc=datetime.now(timezone.utc).isoformat(),
              rules_snapshot=RULES,rules_sha256=hashlib.sha256(rules_snapshot.encode()).hexdigest(),
              annotation_status='designer_reviewed_tile',whole_image_production_approved=False,
              training_status='collected_only_not_trained',
              inputs=dict(sketch='sketch.png',ai_draft='before.png',initial_ai_draft='initial_ai_draft.png'),target='after.png',
              changed_mask='changed_mask.png',loss_mask='core_loss_mask.png',
              review_attestations={k:review[k] for k in required},unresolved=review['unresolved'])
    parent=directory/'correction_examples'/key;parent.mkdir(parents=True,exist_ok=True);out=parent/revision
    if out.exists():
        existing=json.loads((out/'example.json').read_text())
        # Versioned metadata and images must not be silently replaced.
        for field in ['after_rgb_sha256','before_rgb_sha256','sketch_rgb_sha256','full_sketch_rgb_sha256','design_group_id','reviewed_by','rules_corrected','correction_notes','draft_generator_model','ai_proposal_revision']:
            if existing[field]!=data[field]:raise ValueError('Captured revision differs from its immutable review record.')
        return out
    staged=Path(tempfile.mkdtemp(prefix='.capture-',dir=parent))
    try:
        sketch.save(staged/'sketch.png');before.save(staged/'before.png');initial.save(staged/'initial_ai_draft.png');proposal.save(staged/'after.png')
        Image.fromarray(changed.astype(np.uint8)*255).save(staged/'changed_mask.png')
        Image.fromarray(core).save(staged/'core_loss_mask.png')
        (staged/'example.json').write_text(json.dumps(data,indent=2))
        staged.rename(out)
    except Exception:
        shutil.rmtree(staged,ignore_errors=True);raise
    return out


def export_dataset(directory,destination,include_unchanged=False):
    """Export latest eligible revision per tile; preserve every revision in run."""
    directory=Path(directory);destination=Path(destination)
    if destination.exists() or destination.with_suffix('.zip').exists():raise ValueError('Use a new export directory; prior exports are retained.')
    manifest,_=read(directory);selected=[];skipped=[];groups=set()
    for tile in manifest['tiles']:
        folder=directory/'proposals'/tile['id']
        records=sorted((p for p in folder.glob('*.json') if p.stem.isdigit() and p.with_suffix('.png').is_file()),key=lambda p:int(p.stem))
        if not records:continue
        record=records[-1]
        try:sample=capture(directory,tile['id'],record)
        except ValueError as exc:
            skipped.append(dict(tile=tile['id'],reason=str(exc)));continue
        data=json.loads((sample/'example.json').read_text())
        if data['changed_pixels']==0 and not include_unchanged:
            skipped.append(dict(tile=tile['id'],reason='Unchanged example excluded by default.'));continue
        groups.add(data['design_group_id']);selected.append((sample,data))
    if len(groups)>1:raise ValueError('One original design has inconsistent group IDs across tiles; fix metadata before exporting.')
    if not selected:raise ValueError('No eligible reviewed correction examples. Save designer corrections with group/rule notes first.')
    destination.mkdir(parents=True);(destination/'examples').mkdir()
    rows=[]
    for folder,data in selected:
        relative=Path('examples')/data['sample_id'];shutil.copytree(folder,destination/relative)
        rows.append(dict(sample_id=data['sample_id'],design_group_id=data['design_group_id'],
                         example_kind=data['example_kind'],metadata=str(relative/'example.json'),
                         sketch=str(relative/'sketch.png'),ai_draft=str(relative/'before.png'),
                         designer_target=str(relative/'after.png'),loss_mask=str(relative/'core_loss_mask.png'),
                         changed_mask=str(relative/'changed_mask.png')))
    (destination/'dataset.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    report=dict(schema_version=1,samples=len(rows),design_groups=len(groups),group_ids=sorted(groups),
                export_policy='Latest reviewed revision per tile; no-change examples excluded unless requested.',
                skipped=skipped,provider_format='neutral_image_edit_pairs_not_a_provider_finetuning_upload',
                training_status='not_trained',split_policy='Keep all tiles, revisions, variants and crops sharing a design_group_id in the same split.')
    (destination/'dataset_report.json').write_text(json.dumps(report,indent=2))
    (destination/'README.md').write_text('''# Designer correction dataset\n\nCollected examples only: no model has been trained. Each record includes original sketch context, original AI draft context, designer target, changed-pixel mask and a core loss mask. Halo pixels are context, not independent supervision.\n\nThe latest approved revision per tile is exported; historical revisions stay in the review run. Keep all patches, variants and revisions of an original design in one split. Verify group IDs with the designer; filenames/hashes do not establish related-design ancestry. Tile review is not whole-image production approval.\n\nThis is provider-neutral data, not a claimed OpenAI/Sunburst fine-tuning format. Model support and training interfaces must be checked before adapting it for training. A subscription or editing session does not update model weights. Evaluate improvements on held-out whole designs.\n''')
    archive=destination.with_suffix('.zip')
    if archive.exists():raise ValueError('Export ZIP already exists; choose a new destination.')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(destination.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(destination))
    return report,str(archive)
