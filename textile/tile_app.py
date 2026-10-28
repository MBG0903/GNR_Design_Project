"""Human tile review UI. No automatic staircase correction is claimed."""
from pathlib import Path
import json
import numpy as np
from PIL import Image,ImageDraw
import gradio as gr
from .core import exact_palette
from .tiles import read,_load,save_proposal,merge_saved


def build_tile_app(directory):
    directory=Path(directory);manifest,base=read(directory)
    lookup={t['id']:t for t in manifest['tiles']};first=next(iter(lookup))
    def value(im):
        a=np.asarray(im);return dict(background=a,layers=[],composite=a)
    def load(key):
        tile=lookup[key];draft=base.crop(tile['context'])
        sketch=_load(directory/'sketch.png').crop(tile['context'])
        guide=draft.resize((draft.width*3,draft.height*3),Image.Resampling.NEAREST)
        d=ImageDraw.Draw(guide);x0,y0,x1,y1=tile['core_local']
        d.rectangle((x0*3,y0*3,x1*3-1,y1*3-1),outline='#0066ff',width=2)
        info=f"{key}: global core {tile['core']}; full context {tile['context']}; editable local core {tile['core_local']}."
        return sketch,value(draft),guide,info,False,False,False,key
    with gr.Blocks(title='Textile tile staircase review') as demo:
        gr.Markdown('''# Tile staircase review — experiment
Review full front/back curves and tips using the overlap and full-image view.
Blue core boundaries appear only in the separate guide. Save rejects any edits in the context border.
The earlier rejected strip-fitting method remains disabled. This editor does not enforce semantic pixel hierarchy automatically.''')
        with gr.Accordion('Full design context',open=False):
            gr.Image(value=base,type='pil',interactive=False,label='Original dense full design')
        loaded_key=gr.State()
        choice=gr.Dropdown(list(lookup),value=first,label='Tile')
        info=gr.Textbox(interactive=False,label='Native coordinates')
        with gr.Row():
            source=gr.Image(type='pil',interactive=False,label='Sketch context')
            guide=gr.Image(type='pil',interactive=False,label='Editable core boundary (blue guide only)')
        editor=gr.ImageEditor(type='numpy',image_mode='RGB',format='png',label='Native tile pixels; red/white brush',
                              brush=gr.Brush(colors=['#ff0000','#ffffff'],color_mode='fixed',default_size=1),
                              eraser=False,transforms=(),layers=False,fixed_canvas=False)
        normalize=gr.Button('Normalize editor colours to exact red / white')
        gr.Markdown('''Hierarchy: inner/front anchor shorter than outer/back; 3→2→1 with n3<n2<n1,
or 2→1 with n2<n1 where size/shape prevents 3→2→1. Preserve two-pixel gaps and intentional white.
If the complete affected curve is not visible, expand the context or inspect the full component before editing.
Changes crossing a core border need coordinated adjacent-core edits and full-curve review after merging.''')
        reviewer=gr.Textbox(label='Reviewer')
        complete=gr.Checkbox(label='Complete affected front/back curves and tips inspected')
        hierarchy=gr.Checkbox(label='Staircase counts and front/back anchors reviewed')
        gap=gr.Checkbox(label='Two-pixel separation and intentional white reviewed')
        pending=gr.Textbox(label='Unresolved issues (must be empty before saving)',lines=2)
        save=gr.Button('Save versioned tile proposal')
        status=gr.JSON(label='Proposal / merge record')
        merge=gr.Button('Merge saved tile proposals into a REVIEW DRAFT')
        downloads=gr.File(file_count='multiple',label='Merged draft downloads')
        gr.Markdown('''Merge checks size, palette and ownership. Checkboxes record your review; they are not machine proof.
Whole-image curve and seam review is still mandatory. The merged result is not production approval.''')
        def normalize_value(v):
            if not v or v.get('composite') is None:raise gr.Error('Load a tile first.')
            return value(exact_palette(v['composite']))
        def save_value(key,loaded,v,name,c,h,g,unresolved):
            if key!=loaded:raise gr.Error('Wait for the selected tile to finish loading.')
            if not v or v.get('composite') is None:raise gr.Error('Load a tile first.')
            review=dict(reviewer=name,complete_front_back_and_tips_reviewed=c,hierarchy_reviewed=h,
                        two_pixel_gaps_reviewed=g,unresolved=[unresolved.strip()] if unresolved.strip() else [])
            try:path=save_proposal(directory,key,v['composite'],review)
            except ValueError as exc:raise gr.Error(str(exc)) from exc
            return dict(saved=path,production_approved=False)
        def merged():
            try:im,report=merge_saved(directory)
            except ValueError as exc:raise gr.Error(str(exc)) from exc
            folder=directory/'merged_versions';folder.mkdir(exist_ok=True)
            versions=[int(p.name) for p in folder.iterdir() if p.is_dir() and p.name.isdigit()]
            out=folder/f'{max(versions,default=0)+1:04d}';out.mkdir()
            im.save(out/'merged.png');im.save(out/'merged.bmp')
            (out/'review.json').write_text(json.dumps(report,indent=2))
            return report,[str(out/'merged.png'),str(out/'merged.bmp'),str(out/'review.json')]
        choice.change(load,choice,[source,editor,guide,info,complete,hierarchy,gap,loaded_key])
        demo.load(load,choice,[source,editor,guide,info,complete,hierarchy,gap,loaded_key])
        editor.change(lambda:(False,False,False),outputs=[complete,hierarchy,gap])
        normalize.click(normalize_value,editor,editor)
        save.click(save_value,[choice,loaded_key,editor,reviewer,complete,hierarchy,gap,pending],status)
        merge.click(merged,outputs=[status,downloads])
    return demo
