"""Designer tile editor and correction-data collection; no automatic training."""
from pathlib import Path
import json
import numpy as np
from PIL import Image,ImageDraw
import gradio as gr
from .core import RULES,exact_palette
from .tiles import read,_load,save_proposal,merge_saved,paint_pixel
from .correction_data import validate_annotation,capture,export_dataset


def build_tile_app(directory,design_group_id='',draft_generator_model='unknown',ai_client=None,image_model='gpt-image-2.5-sunburst'):
    directory=Path(directory);manifest,base=read(directory)
    lookup={t['id']:t for t in manifest['tiles']};first=next(iter(lookup))
    def value(im):
        a=np.asarray(im);return dict(background=a,layers=[],composite=a)
    def load(key):
        tile=lookup[key];original=base.crop(tile['context']);draft=original.copy();notes='';tags=[];ai_revision=None;before_view=original
        folder=directory/'proposals'/key
        records=sorted((p for p in folder.glob('*.json') if p.stem.isdigit() and p.with_suffix('.png').is_file()),key=lambda p:int(p.stem))
        if records:
            record=records[-1];draft=_load(record.with_suffix('.png'));review=json.loads(record.read_text())
            notes=review.get('correction_notes','');tags=review.get('rules_corrected',[]);ai_revision=review.get('ai_proposal_revision')
            if ai_revision is not None:
                from .tile_genai import proposal_input
                before_view,_=proposal_input(directory,key,ai_revision,manifest)
        else:
            from .tile_batch import _latest
            from .tile_genai import proposal_input
            ai_revision=_latest(directory,key)
            if ai_revision is not None:
                draft,_=proposal_input(directory,key,ai_revision,manifest)
                before_view=draft.copy()
        sketch=_load(directory/'sketch.png').crop(tile['context'])
        guide=original.resize((original.width*3,original.height*3),Image.Resampling.NEAREST)
        d=ImageDraw.Draw(guide);x0,y0,x1,y1=tile['core_local']
        d.rectangle((x0*3,y0*3,x1*3-1,y1*3-1),outline='#0066ff',width=2)
        info=f"{key}: global core {tile['core']}; full context {tile['context']}; editable local core {tile['core_local']}."
        return sketch,value(draft),guide,info,False,False,False,key,before_view,notes,tags,ai_revision
    def preview(v,z):
        if not v or v.get('composite') is None:return None,int(z)
        im=_load(v['composite']);z=int(z)
        grid=im.resize((im.width*z,im.height*z),Image.Resampling.NEAREST);d=ImageDraw.Draw(grid)
        if z>=4:
            for x in range(0,grid.width,z):d.line((x,0,x,grid.height-1),fill='#b0b0b0')
            for y in range(0,grid.height,z):d.line((0,y,grid.width-1,y),fill='#b0b0b0')
        return grid,z
    with gr.Blocks(title='Designer textile correction panel') as demo:
        gr.Markdown('''# Designer correction panel
Edit the AI proposal, explain the correction and save a reviewed revision.
Every save captures a paired example; model weights are **not** updated here.
Context borders are read-only. Inspect complete front/back curves and tips before signing off.''')
        with gr.Accordion('Full design context',open=False):
            gr.Image(value=base,type='pil',format='png',interactive=False,label='Original full AI design')
        group=gr.Textbox(value=design_group_id,label='Original design/group ID — same ID for all its variants and crops')
        model=gr.Textbox(value=draft_generator_model,label='AI generator model ID (use unknown if unverified; never enter an API key)')
        loaded_key=gr.State();grid_zoom=gr.State(8);pending_ai_revision=gr.State(None)
        choice=gr.Dropdown(list(lookup),value=first,label='Tile')
        info=gr.Textbox(interactive=False,label='Native coordinates')
        with gr.Row():
            source=gr.Image(type='pil',format='png',interactive=False,label='Artist sketch')
            original_view=gr.Image(type='pil',format='png',interactive=False,label='AI proposal before designer edits (training input)')
            guide=gr.Image(type='pil',format='png',interactive=False,label='Editable core guide (blue is not saved)')
        editor=gr.ImageEditor(type='numpy',image_mode='RGB',format='png',label='Designer target — brush editor',
                              brush=gr.Brush(colors=['#ff0000','#ffffff'],color_mode='fixed',default_size=1),
                              eraser=False,transforms=(),layers=False,fixed_canvas=False)
        propose_ai=gr.Button('Request Sunburst tile proposal — paid API',interactive=ai_client is not None)
        gr.Markdown('Sunburst proposals need review; generated cores are clipped to ownership and are not auto-merged. Configure the optional API client in Colab to enable this button.')
        normalize=gr.Button('Normalize brush colours to exact red / white')
        with gr.Accordion('Precise one-pixel editing',open=True):
            colour=gr.Radio(['red','white'],value='red',label='Click colour')
            zoom=gr.Slider(2,16,value=8,step=1,label='Integer pixel magnification')
            pixel_grid=gr.Image(type='pil',format='png',interactive=False,label='Click a cell to set exactly one native pixel; use fullscreen for detail',height=650)
        gr.Markdown('''Inspect inner/front versus outer/back anchors and every tip/transition.
Prefer 3→2→1 with n3<n2<n1; allow 2→1 with n2<n1 when shape/size prevents 3→2→1.
Preserve two-pixel gaps and deliberate white. Review the full component if a curve leaves the context.''')
        reviewer=gr.Textbox(label='Designer / reviewer name')
        rules=gr.Dropdown([(rule,rule.split(':',1)[0]) for rule in RULES],multiselect=True,label='Rules corrected in this revision')
        notes=gr.Textbox(label='What was wrong, and how did you correct it?',lines=3)
        complete=gr.Checkbox(label='Complete affected front/back curves and tips inspected')
        hierarchy=gr.Checkbox(label='Staircase counts and front/back anchors reviewed')
        gap=gr.Checkbox(label='Two-pixel separation and intentional white reviewed')
        pending=gr.Textbox(label='Unresolved issues (clear after resolving)',lines=2)
        save=gr.Button('Save designer correction + training example')
        status=gr.JSON(label='Save / collection / merge status')
        merge=gr.Button('Merge saved tiles into a REVIEW DRAFT')
        downloads=gr.File(file_count='multiple',label='Merged draft downloads')
        include_no_change=gr.Checkbox(False,label='Include designer-approved unchanged tiles in dataset export')
        export=gr.Button('Export correction dataset ZIP — no training is run')
        dataset_download=gr.File(label='Correction dataset download')
        gr.Markdown('''Captured pairs contain the original sketch, original AI draft, designer target, changed-pixel mask,
core supervision mask, rule notes and review metadata. Only the latest reviewed revision per tile is exported by default;
all saved revisions stay in the run. Keep all patches/variants of one design in the same train/validation/test split.
Checkboxes record human review; they do not prove semantic compliance or whole-image production readiness.''')
        def normalize_value(v):
            if not v or v.get('composite') is None:raise gr.Error('Load a tile first.')
            return value(exact_palette(v['composite']))
        def refresh(v,z):
            grid,actual=preview(v,z);return grid,actual,False,False,False
        def click_pixel(key,loaded,v,c,z,event:gr.SelectData):
            if key!=loaded:raise gr.Error('Wait for the selected tile to load.')
            if not v or v.get('composite') is None:raise gr.Error('Load a tile first.')
            try:im=paint_pixel(v['composite'],lookup[key],list(event.index),int(z),c)
            except ValueError as exc:raise gr.Error(str(exc)) from exc
            return value(im)
        def save_value(key,loaded,v,name,c,h,g,unresolved,group_id,model_id,tags,explanation,ai_revision):
            if key!=loaded:raise gr.Error('Wait for the selected tile to finish loading.')
            if not v or v.get('composite') is None:raise gr.Error('Load a tile first.')
            review=dict(reviewer=name,complete_front_back_and_tips_reviewed=c,hierarchy_reviewed=h,
                        two_pixel_gaps_reviewed=g,unresolved=[unresolved.strip()] if unresolved.strip() else [],
                        design_group_id=group_id,draft_generator_model=model_id or 'unknown',
                        rules_corrected=tags or [],correction_notes=explanation,ai_proposal_revision=ai_revision)
            tile_before=base.crop(lookup[key]['context'])
            changed=not np.array_equal(np.asarray(tile_before),np.asarray(v['composite']))
            try:
                validate_annotation(review,changed)
                path=Path(save_proposal(directory,key,v['composite'],review))
                example=capture(directory,key,path.with_suffix('.json'))
            except ValueError as exc:raise gr.Error(str(exc)) from exc
            metadata=json.loads((example/'example.json').read_text())
            return dict(saved=str(path),training_example=str(example),changed_pixels=metadata['changed_pixels'],
                        training_status='collected_only_not_trained',production_approved=False)
        def generate_value(key,loaded,v):
            if key!=loaded or not v or v.get('composite') is None:
                raise gr.Error('Load the selected tile first.')
            from .tile_genai import propose
            try:candidate,metadata=propose(directory,key,v['composite'],ai_client,model=image_model)
            except ValueError as exc:raise gr.Error(str(exc)) from exc
            return value(candidate),candidate,metadata['revision'],image_model,metadata,False,False,False,[],''
        def merged():
            try:im,report=merge_saved(directory)
            except ValueError as exc:raise gr.Error(str(exc)) from exc
            folder=directory/'merged_versions';folder.mkdir(exist_ok=True)
            versions=[int(p.name) for p in folder.iterdir() if p.is_dir() and p.name.isdigit()]
            out=folder/f'{max(versions,default=0)+1:04d}';out.mkdir()
            im.save(out/'merged.png');im.save(out/'merged.bmp');(out/'review.json').write_text(json.dumps(report,indent=2))
            return report,[str(out/'merged.png'),str(out/'merged.bmp'),str(out/'review.json')]
        def export_training(include):
            folder=directory/'dataset_exports';folder.mkdir(exist_ok=True)
            versions=[int(p.stem) for p in folder.iterdir() if p.stem.isdigit()]
            out=folder/f'{max(versions,default=0)+1:04d}'
            try:report,archive=export_dataset(directory,out,include_unchanged=include)
            except ValueError as exc:raise gr.Error(str(exc)) from exc
            return report,archive
        outputs=[source,editor,guide,info,complete,hierarchy,gap,loaded_key,original_view,notes,rules,pending_ai_revision]
        choice.change(load,choice,outputs);demo.load(load,choice,outputs)
        editor.change(refresh,[editor,zoom],[pixel_grid,grid_zoom,complete,hierarchy,gap])
        zoom.change(preview,[editor,zoom],[pixel_grid,grid_zoom])
        pixel_grid.select(click_pixel,[choice,loaded_key,editor,colour,grid_zoom],editor)
        normalize.click(normalize_value,editor,editor)
        save.click(save_value,[choice,loaded_key,editor,reviewer,complete,hierarchy,gap,pending,group,model,rules,notes,pending_ai_revision],status)
        propose_ai.click(generate_value,[choice,loaded_key,editor],[editor,original_view,pending_ai_revision,model,status,complete,hierarchy,gap,rules,notes])
        merge.click(merged,outputs=[status,downloads])
        export.click(export_training,include_no_change,outputs=[status,dataset_download])
    return demo
