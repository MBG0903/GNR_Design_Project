"""Update the uploaded v4 notebook without executing its code or retaining outputs."""
import argparse
import ast
import base64
import hashlib
import json
from pathlib import Path
import zlib

ROOT=Path(__file__).resolve().parents[1]


def replace_nodes(source,names,replacements):
    lines=source.splitlines(True);tree=ast.parse(source)
    nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.Assign)) and
           ((isinstance(n,ast.FunctionDef) and n.name in names) or
            (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in names for t in n.targets)))]
    for node in sorted(nodes,key=lambda n:n.lineno,reverse=True):
        key=node.name if isinstance(node,ast.FunctionDef) else next(t.id for t in node.targets if isinstance(t,ast.Name) and t.id in names)
        lines[node.lineno-1:node.end_lineno]=[replacements.get(key,'')+'\n']
    return ''.join(lines)


def wrap(source,flag,label):
    return f'if {flag}:\n'+''.join('    '+line if line.strip() else line for line in source.splitlines(True))+f'\nelse:\n    print({label!r})\n'


def code(source):return dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=source.splitlines(True))

def md(source):return dict(cell_type='markdown',metadata={},source=source.splitlines(True))


def build(source_path,destination):
    original=json.loads(Path(source_path).read_text())
    cells=original['cells']
    if len(cells)!=31 or 'designer_profile' not in ''.join(cells[5]['source']):
        raise ValueError('This updater expects the reviewed 31-cell v4 notebook; source differs.')
    sources={i:''.join(c['source']) for i,c in enumerate(cells)}
    core=(ROOT/'textile/staircase_v5.py').read_text()
    digest=hashlib.sha256(core.encode()).hexdigest()
    # Revised prompt follows the latest diagram, not the superseded run-count rule.
    prompt='''PHOTOSHOP_PROMPT = f"""Convert the supplied Step_2 sketch into a dense textile Step_3 design. Preserve motifs, positions, proportions, borders and meaningful white spaces. Ignore pencil/grid noise. Canvas: exactly {W} x {H} native pixels.
Use only pure red #FF0000 and white #FFFFFF. Fill intended solids red; preserve intentional white channels, cut-outs, veins and eyes. Keep distinct pieces separated by two native white pixels at their closest approach; preserve wider background spaces and genuinely continuous stems/curls.
Construct applicable curved corners from two flat anchors A and B through a shared one-pixel boundary transition M. Flat lengths are free. From either flat toward M, contour run lengths start at 2, 3 or 4, hold or decrease one level at a time, and end at 1. Example: horizontal 3,2,2 -> M(1 pixel) -> vertical 2,2,3; reading either half toward M gives 3,2,2,1. No skipping levels or growing toward M. There is no requirement about numbers of steps at each level. Apply in all orientations while preserving the shape; mirror only intended symmetry.
Keep equivalent repeated pieces consistent. Use crisp edges without antialiasing, bumps, stray pixels, gradients, textures or transparency. Preserve motif identity when uncertain. Export PNG/BMP at the original dimensions."""'''
    sources[2]=replace_nodes(sources[2],{'PHOTOSHOP_PROMPT'},{'PHOTOSHOP_PROMPT':prompt})
    # Keep the visible helper functions, replacing only old staircase definitions.
    obsolete={'groups','extreme_lines','lead_steps','diagonal_steps','curve_ok','curve_rule_pct'}
    sources[3]=replace_nodes(sources[3],obsolete,{})
    insert='''# Updated A/M/B geometry is loaded from the embedded v5 runtime.
V5_CORE.install_bindings(globals())
'''
    sources[3]=sources[3].replace('# ---------------- measures ----------------','# ---------------- measures ----------------\n'+insert)
    # Reject hidden scaling of the input and evaluation truth.
    strict='''def import_export(path, size):
    """Project palette at the exact native size; never silently resize geometry."""
    with Image.open(path) as opened:
        if opened.size != tuple(size):
            raise ValueError(f"Export size {{opened.size}} differs from Step_2 {{tuple(size)}}. Supply a registered native-size export; v5 refuses automatic resizing.")
        soft = redness_of(opened)
    return soft >= 0.5, soft, {{"export_size": tuple(size), "note": "native size, no resize"}}
'''.replace('{{','{').replace('}}','}')
    sources[3]=replace_nodes(sources[3],{'import_export'},{'import_export':strict})
    # A missing inbox draft is an input handoff, not a geometry failure.
    old_missing='''inbox = inbox_files()
if DESIGN not in inbox:
    raise FileNotFoundError(f"No Photoshop export for design {DESIGN} in {INBOX}. Upload it as '{DESIGN} GNR 3.png'.")'''
    assert old_missing in sources[3]
    sources[3]=sources[3].replace(old_missing,'''# Optional: paste the FULL Drive path to an existing AI-generated Step_3 draft.
# Leave blank to reuse the inbox draft, or upload one when it is missing.
AI_DRAFT_PATH = ""
inbox = inbox_files()
if AI_DRAFT_PATH.strip():
    selected_path = Path(AI_DRAFT_PATH.strip()).expanduser()
    if not selected_path.is_file():
        raise FileNotFoundError(f"AI_DRAFT_PATH does not exist: {selected_path}")
    inbox[DESIGN] = V5_CORE.stage_ai_draft(selected_path.read_bytes(), INBOX, DESIGN, (W, H))
elif DESIGN not in inbox:
    print(f"Step_2 for design {DESIGN} is loaded ({W} x {H}).")
    print("This finisher needs an AI-generated Step_3 draft before it can edit the pixels.")
    print("Choose your Photoshop/Sunburst draft below, not the Step_2 sketch or designer reference.")
    print("If no draft exists yet, generate it using Cell 2's prompt, then rerun this cell.")
    from google.colab import files
    uploaded_drafts = files.upload()
    if len(uploaded_drafts) != 1:
        raise ValueError("Select exactly one AI Step_3 draft. You can also set AI_DRAFT_PATH above, then rerun this cell.")
    draft_bytes = next(iter(uploaded_drafts.values()))
    inbox[DESIGN] = V5_CORE.stage_ai_draft(draft_bytes, INBOX, DESIGN, (W, H))
print("Using AI draft:", inbox[DESIGN])''')
    old='REF = redness_of(ref_img.resize((W, H), Image.NEAREST) if ref_img.size != (W, H) else ref_img) >= 0.5'
    assert old in sources[3]
    sources[3]=sources[3].replace(old,'''if ref_img.size != (W, H):
        raise ValueError("Designer reference differs from Step_2 dimensions; register it explicitly before comparison.")
    REF = redness_of(ref_img) >= 0.5''')
    sources[2]='V5_ACTIVE_INPUT = None  # A changed sketch invalidates older raster state.\n'+sources[2]
    sources[3]='V5_ACTIVE_INPUT = None  # Remains invalid if import or reporting fails.\nV5_CORE.check_selected_sketch(globals())\n'+sources[3]
    sources[3]+='\nV5_ACTIVE_INPUT = V5_CORE.capture_input_context(globals(), inbox[DESIGN])\nprint("Input ready for design", DESIGN, "— continue to Cell 4.")\n'
    # Replace the old constructor/repair engine rather than leaving competing definitions.
    run_at=sources[5].index('# ---------------- run ----------------')
    sources[5]='''# CELL 5: A/M/B CURVE CONSTRUCTION — updated to the supplied staircase diagram.
MAX_SHIFT = 2        # maximum native Chebyshev boundary displacement
MAX_REPEAT = 8       # bounded search, per 2/3/4 level; NOT a step-count hierarchy
V5_MAX_CORNER_EDITS = 250
V5_PROTECTED_WHITE = globals().get('V5_PROTECTED_WHITE', None)  # optional explicit native bool mask
SKIPPED = {}
V5_CORE.install_bindings(globals())
# constructor: V5_CORE.construct_corner(); validator: V5_CORE.corner_check().
# Unsupported junctions/ambiguous anchors remain unresolved for the designer.

'''+sources[5][run_at:]
    # Final cleanup can only replace the candidate when complete affected contours pass.
    sources[10]=replace_nodes(sources[10],{'save_production'},{'save_production':'# save_production is the native-size checked REVIEW exporter bound below.\nV5_CORE.install_bindings(globals())'})
    old='''RED_GUARD, guard_info = separation_guard(RED_LEARN, RED_PS, PROB_LEARN)
RED_TIDY = tidy(RED_GUARD, np.ones_like(RED_GUARD))'''
    assert old in sources[10]
    sources[10]=sources[10].replace(old,'''if V5_USE_LEARNED_FINISHER:
    RED_GUARD, guard_info = separation_guard(RED_LEARN, RED_PS, PROB_LEARN)
else:
    RED_GUARD = globals().get("RED_FINAL", RED_PS).copy()
    guard_info = {"mode": "geometry-only review; learned finisher disabled"}
RED_TIDY = RED_GUARD.copy()
tidy_candidate = tidy(RED_GUARD, np.ones_like(RED_GUARD))
tidy_ok, tidy_report = V5_CORE.transition_guard(RED_GUARD, tidy_candidate, V5_PROTECTED_WHITE)
if tidy_ok:
    RED_TIDY = tidy_candidate
else:
    print("Pre-construction cleanup retained for review:", tidy_report["reason"])''')
    sources[10]=sources[10].replace('RED_DONE = tidy(RED_DONE, np.ones_like(RED_DONE))','''post_candidate = tidy(RED_DONE, np.ones_like(RED_DONE))
post_ok, post_report = V5_CORE.transition_guard(RED_DONE, post_candidate, V5_PROTECTED_WHITE)
if post_ok:
    RED_DONE = post_candidate
else:
    print("Post-construction cleanup rejected:", post_report["reason"])''')
    sources[10]=sources[10].replace('cols += [("learned", RED_LEARN), ("FINAL", RED_DONE)]','''if V5_USE_LEARNED_FINISHER:
    cols.append(("learned", RED_LEARN))
cols.append(("FINAL", RED_DONE))''')
    sources[10]+='''\nprint("V5 export status:", V5_LAST_EXPORT_REPORT["status"])
print("Corners:", {k: V5_LAST_EXPORT_REPORT["curves"][k] for k in ("passed", "failed", "unresolved")})
print("Saved as a review draft; semantic applicability and motif fidelity require designer review.")
'''
    # LLM proposals use the same rule and are checked after seam cleanup.
    sources[11]=sources[11].replace('Rule D: curved edges are smooth pixel staircases: a flat line at the extreme of the curve, then steps of 3, 2, 1 pixels, never jumping back up; no jagged or wobbly edges.',
        'Rule D: an applicable curved corner has flat anchors A/B and a shared 1-pixel boundary transition M. From either flat toward M, run lengths start at 2/3/4, hold or decrease one level at a time to 1. Example horizontal 3,2,2 -> M(1) -> vertical 2,2,3. Flat lengths are free; there is no step-count dominance. Preserve shape in all orientations.')
    old='''    if not joins and not more_bumps:
        RED_REVIEWED = trial'''
    new='''    geometry_ok, geometry_report = V5_CORE.transition_guard(RED_REVIEWED, trial, V5_PROTECTED_WHITE)
    v["geometry_check"] = {"passed": geometry_ok, "reason": geometry_report["reason"]}
    if not joins and not more_bumps and geometry_ok:
        RED_REVIEWED = trial'''
    assert old in sources[11];sources[11]=sources[11].replace(old,new)
    sources[11]=sources[11].replace('v["reason"] += "  [not applied: " + ("would join pieces" if joins else "would add bumps") + "]"',
        'v["reason"] += "  [not applied: " + ("would join pieces" if joins else "would add bumps" if more_bumps else geometry_report["reason"]) + "]"')
    # Package shared compatibility library from pure definitions, no mounts/runs/API calls.
    snippets=[]
    wanted={1:{'pair_files','inbox_files'},3:{'redness_of','import_export','rgb_of','gap_widths','bump_count','shape_match','measure'},
            4:{'nearest_two','bump_map','tidy','centre_points','pen_mask','two_px_separations'},
            9:{'block','make_unet','run_unet','finish_learned','gap_rows'},10:{'separation_guard'}}
    for i,names in wanted.items():
        tree=ast.parse(sources[i])
        snippets.extend(ast.get_source_segment(sources[i],n) for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names)
    pattern=next(ast.get_source_segment(sources[1],n) for n in ast.parse(sources[1]).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='NAME_PATTERN' for t in n.targets))
    compat='''# Generated v5 compatibility library. Original gnr_v4_lib.py is preserved.
import re,json,types
from pathlib import Path
import numpy as np
import cv2
from scipy import ndimage as ndi
from PIL import Image
from skimage.morphology import skeletonize
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    torch=None
_core=types.ModuleType('gnr_staircase_v5_shared')
exec(CORE_SOURCE, _core.__dict__)
N8=_core.N8;N4=_core.N4
DEVICE="cuda" if torch is not None and torch.cuda.is_available() else "cpu"
MAX_GAP=5;UP=4;MIN_THICK=2.0
MAX_SHIFT=globals().get('MAX_SHIFT',2);MAX_REPEAT=globals().get('MAX_REPEAT',8)
V5_MAX_CORNER_EDITS=globals().get('V5_MAX_CORNER_EDITS',250)
V5_PROTECTED_WHITE=globals().get('V5_PROTECTED_WHITE',None)
MODEL_FILE=globals().get('MODEL_FILE',Path(globals().get('PIPE','.'))/'05_Finisher_Model'/'finisher_v2.pt')
'''.replace('exec(CORE_SOURCE, _core.__dict__)',f'exec({core!r}, _core.__dict__)')
    compat+='\n'+pattern+'\n\n'+'\n\n'.join(snippets)+'\n\n_core.install_bindings(globals())\n'
    compat=compat.replace('    m = nn.ModuleDict()', '    if torch is None: raise ImportError("PyTorch is required for the optional saved finisher.")\n    m = nn.ModuleDict()')
    compat=compat.replace('curve rule % (designer ~80)','A/M/B corner screening %')
    # Known batch entry point uses the supplied v5 library. Unknown external scripts are gated separately.
    sources[12]=sources[12].replace('exec(open(PIPE / "Notebook" / "gnr_v4_lib.py", encoding="utf-8").read())',
        'exec(V5_COMPAT_PATH.read_text(encoding="utf-8"))')
    sources[12]=sources[12].replace('red = tidy(red, np.ones_like(red))','''cleanup = tidy(red, np.ones_like(red))
    cleanup_ok, cleanup_check = _core.transition_guard(red, cleanup, V5_PROTECTED_WHITE)
    if cleanup_ok:
        red = cleanup''')
    for i in (4,5,6,9,10,11):
        sources[i]='V5_CORE.require_input_context(globals())\n'+sources[i]
    # Apply self-contained output wording consistently to all visible measure consumers.
    for i in sources:
        sources[i]=sources[i].replace('curve rule % (designer ~80)','A/M/B corner screening %')
    for i,c in enumerate(cells):
        if c['cell_type']=='code':
            c['source']=sources[i].splitlines(True);c['outputs']=[];c['execution_count']=None
    # Expensive/externally delegated experiments are opt-in; normal first run is geometry only.
    for i,flag,label in [(7,'V5_RUN_FINISHER_TRAINING','Synthetic training-data generation skipped.'),
                         (8,'V5_RUN_FINISHER_TRAINING','Finisher training skipped.'),
                         (9,'V5_USE_LEARNED_FINISHER','Saved learned finisher skipped: geometry-only mode.'),
                         (11,'V5_RUN_LLM_REVIEW and V5_USE_LEARNED_FINISHER','Paid LLM review skipped.'),
                         (12,'V5_RUN_LEGACY_BATCH','Optional learned batch skipped; enable only with a saved model.')]:
        cells[i]['source']=wrap(''.join(cells[i]['source']),flag,label).splitlines(True)
    for i in range(13,len(cells)):
        flag='V5_RUN_PAIR_MODEL_EXPERIMENTS' if 14<=i<=18 else 'V5_RUN_EXTERNAL_CELLS'
        cells[i]['source']=wrap(''.join(cells[i]['source']),flag,
            'Optional experiment skipped. External Drive scripts have not been upgraded or certified by this notebook.').splitlines(True)
    def packed(s):return base64.b64encode(zlib.compress(s.encode())).decode()
    core_payload,lib_payload=packed(core),packed(compat)
    lib_hash=hashlib.sha256(compat.encode()).hexdigest()
    bootstrap=f'''# V5 SETUP — only installs missing geometry dependencies; no Drive/API/training calls.
import base64,zlib,sys,subprocess,importlib.util,types
from pathlib import Path
needed={{"numpy":"numpy", "scipy":"scipy", "PIL":"Pillow", "cv2":"opencv-python-headless", "skimage":"scikit-image", "matplotlib":"matplotlib"}}
missing=[package for module,package in needed.items() if importlib.util.find_spec(module) is None]
if missing:subprocess.check_call([sys.executable,"-m","pip","install",*missing])
V5_CORE_SOURCE=zlib.decompress(base64.b64decode({core_payload!r})).decode()
V5_COMPAT_SOURCE=zlib.decompress(base64.b64decode({lib_payload!r})).decode()
V5_CORE=types.ModuleType("gnr_staircase_v5_embedded")
exec(V5_CORE_SOURCE,V5_CORE.__dict__)
V5_RUN_FINISHER_TRAINING=False
V5_USE_LEARNED_FINISHER=False
V5_RUN_LLM_REVIEW=False
V5_RUN_LEGACY_BATCH=False
V5_RUN_PAIR_MODEL_EXPERIMENTS=False
V5_RUN_EXTERNAL_CELLS=False
print("V5 geometry runtime loaded. Default: geometry-only, no training or paid calls.")
'''
    selftest='''# V5 SELF-TEST — safe to run before mounting Drive. No GPU or API key required.
import json
print(json.dumps(V5_CORE.self_test(),indent=2))
'''
    install=f'''# V5 SHARED LIBRARY — new versioned file; original gnr_v4_lib.py is not changed.
V5_COMPAT_PATH=PIPE/"Notebook"/"gnr_v5_lib_{lib_hash[:12]}.py"
V5_COMPAT_PATH.parent.mkdir(parents=True,exist_ok=True)
if V5_COMPAT_PATH.exists() and V5_COMPAT_PATH.read_text()!=V5_COMPAT_SOURCE:
    raise ValueError("Existing versioned v5 library differs; preserve it and use a new version.")
V5_COMPAT_PATH.write_text(V5_COMPAT_SOURCE)
print("Shared v5 batch library:",V5_COMPAT_PATH)
'''
    intro=md('''# GNR Step 3 v5 — A/M/B staircase update
Updated from the supplied v4 notebook using the latest staircase diagram.

**Reference:** flat A → horizontal 3,2,2 → shared M (1 pixel) → vertical 2,2,3 → flat B. From either flat toward M, run lengths start at 2/3/4, hold or decrease one level to 1. Flat lengths are free; there is no step-count dominance. This version requires a single shared transition for the illustrated corner family. Ambiguous junctions and other corner families remain unresolved.

**Start here:** run V5 SETUP and V5 SELF-TEST first (CPU is sufficient). Then run original Cells 1–6 and Cell 10 for one native-size AI-generated Step_3 draft. Set DESIGN in Cell 2. In Cell 3, reuse the inbox file, paste an existing draft's full Drive path into AI_DRAFT_PATH, or upload one when prompted. The finisher requires that draft; it does not generate Step_3 from Step_2 in its default mode. Use Cell 2's Photoshop prompt first if you do not yet have a draft. The geometry-only path does not require retraining or an API key.

**Optional:** enable V5_USE_LEARNED_FINISHER only to load an existing saved finisher and run Cells 9–10. Enable V5_RUN_LLM_REVIEW for paid Cell 11 review. Training, model experiments and external scripts are disabled by default. Unknown external Drive scripts were not provided and are not upgraded by changing this notebook.

The embedded constructor searches bounded native-pixel paths, checks the rendered contour and preserves already-passing corners. Re-extraction follows each accepted edit. Cleanup and LLM replacements must pass checks on complete affected raster regions. Anchors are inferred from orthogonal flat extrema, not from the centre of a crop. This inference is conservative and does not establish semantic identity.

Exports go to **02_Review_Output_v5**, with PNG/BMP and a JSON report of passed, failed and unresolved detected corners and narrow raster-region gaps. Native dimensions are checked against the original Step_2 file. All exports remain review drafts: semantic applicability, intentional white and the completeness of curve detection require designer review. The original notebook and its legacy shared library are preserved.

Old saved outputs were cleared so they cannot be mistaken for measurements from this update. Source-update tests do not establish full-image quality or the 10% manual-work target.
''')
    updated=[intro,code(bootstrap),code(selftest),cells[1],code(install)]+cells[2:]
    for i,cell in enumerate(updated):
        cell['id']=f'v5-{i:02d}-'+hashlib.sha256(''.join(cell['source']).encode()).hexdigest()[:8]
        cell.setdefault('metadata',{}).pop('executionInfo',None)
        cell['metadata'].pop('outputId',None)
        for key in ('base_uri','height'):
            cell['metadata'].get('colab',{}).pop(key,None)
    original['cells']=updated
    original['nbformat']=4
    original['nbformat_minor']=5
    original['metadata'].pop('widgets',None)
    original['metadata'].setdefault('colab',{})['name']='GNR_Step3_v5_AMB_Staircase.ipynb'
    original['metadata']['gnr_v5_update']={'core_sha256':digest,'source_notebook':Path(source_path).name,
        'original_notebook_sha256':hashlib.sha256(Path(source_path).read_bytes()).hexdigest(),
        'default_mode':'geometry_only_review','training_or_api_executed_by_updater':False}
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(original,indent=1))
    return destination


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('source');parser.add_argument('--output',default=str(ROOT/'notebooks/GNR_Step3_v5_AMB_Staircase.ipynb'))
    args=parser.parse_args();print(build(args.source,args.output))
