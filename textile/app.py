"""Gradio editor, usable locally or inline in Colab."""
import argparse
import tempfile
import numpy as np
import gradio as gr
from .core import RULES, rgb, exact_palette, validate, export_reviewed


def editor_value(im):
    return {'background': np.asarray(im), 'layers': [], 'composite': np.asarray(im)}


def composite(value):
    if value is None or value.get('composite') is None:
        raise gr.Error('Load a GenAI draft first.')
    return value['composite']


def build_app():
    with gr.Blocks(title='Sketch to textile raster') as demo:
        gr.Markdown('''# Sketch → textile raster
Load the Step 2 sketch and its GenAI proposal, then inspect and edit with red/white brushes.
The editor checks raster constraints; the vision agent and designer review semantic rules.
Exports are reviewed rasters; loom-specific production approval is a separate step.''')
        source_state = gr.State()
        source = gr.Image(type='pil', label='Step 2 sketch (native pixels)', image_mode='RGB')
        model_draft = gr.Image(type='pil', label='Optional: upload model-generated draft PNG', image_mode='RGB')
        load_draft = gr.Button('Load model draft for current Step 2 sketch')
        editor = gr.ImageEditor(type='numpy', image_mode='RGB', format='png',
                                label='Draft editor: zoom for pixel inspection; use red/white brush',
                                brush=gr.Brush(colors=['#ff0000', '#ffffff'], color_mode='fixed', default_size=1),
                                eraser=False, transforms=(), layers=False, canvas_size=(401, 340), fixed_canvas=False)
        normalize = gr.Button('Apply exact red/white palette to edited pixels')
        check = gr.Button('Check palette and dimensions')
        diagnostics = gr.JSON(label='Automated checks (semantic rules need review)')
        uncertainty = gr.Textbox(label='Unresolved regions / questions (clear only after reviewing)',
                                 value='Review all motifs, piece separation, intentional white channels, tapers and repeated families.', lines=3)
        rules = gr.CheckboxGroup(RULES, label='Designer sign-off: inspect at native pixel resolution')
        reviewer = gr.Textbox(label='Reviewer name')
        export = gr.Button('Export reviewed PNG + BMP + review record')
        files = gr.File(file_count='multiple', label='Downloads')
        gr.Markdown('''The source image is retained separately. Loading a new draft resets sign-off.
Any editor change resets sign-off and invalidates previous download links; downloaded files remain snapshots.
No resizing or motif relocation is applied. Do not use browser-scaled previews to judge one-pixel gaps.''')

        def use_model_draft(im, draft):
            if im is None or draft is None:
                raise gr.Error('Upload both the original sketch and the model draft.')
            if im.size != draft.size:
                raise gr.Error('Model draft dimensions must match the original sketch.')
            return rgb(im), editor_value(exact_palette(draft)), [], None, 'Review model errors, white channels, separation, tapers, repeated families and uncertainty map.'
        load_draft.click(use_model_draft, [source, model_draft],
                         [source_state, editor, rules, files, uncertainty])
        source.change(lambda: (None, None, [], None), outputs=[source_state, editor, rules, files])
        editor.change(lambda: ([], None), outputs=[rules, files])
        normalize.click(lambda v: editor_value(exact_palette(composite(v))), editor, editor)

        def inspect(v, original):
            if original is None:
                raise gr.Error('Load a GenAI draft for the current source first.')
            return validate(composite(v), original.size)
        check.click(inspect, [editor, source_state], diagnostics)

        def save(v, original, approved, name, pending):
            if original is None:
                raise gr.Error('Load a GenAI draft first.')
            try:
                return export_reviewed(composite(v), original, approved, name, pending,
                                       tempfile.mkdtemp(prefix='textile-export-'))
            except ValueError as exc:
                raise gr.Error(str(exc)) from exc
        export.click(save, [editor, source_state, rules, reviewer, uncertainty], files)
    return demo


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=7860)
    args = parser.parse_args()
    build_app().launch(server_name='127.0.0.1', server_port=args.port)
