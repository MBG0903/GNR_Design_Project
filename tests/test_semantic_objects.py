import ast
import base64
import io
import json
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace
import unittest
import zipfile

import numpy as np
from PIL import Image,ImageDraw

from textile.components import digest
from textile.semantic_objects import (build_object_catalog,list_objects,import_reviews,
    export_current_approved,validate_proposals,views,catalog_summary,add_manual_candidate)
from textile.object_review import write_review_gallery,_runs


class Vision:
    def __init__(self, plan=None):
        self.responses=self;self.calls=[]
        self.plan=plan or {'objects':[
            {'id':'animal','category':'peacock','label':'complete peacock',
             'bbox':[1,1,13,10],'uncertain':False},
            {'id':'feather','category':'feather','label':'tail feather',
             'bbox':[8,3,4,5],'parent_id':'animal','uncertain':False}],
             'unresolved':['tiny ornament needs review']}
    def create(self,**kw):
        self.calls.append(kw)
        return SimpleNamespace(output_text=json.dumps(self.plan))


class Segmenter:
    signature='fixture-mask-backend'
    def __init__(self):self.calls=0
    def __call__(self,image,box):
        self.calls+=1
        return np.ones((image.height,image.width),bool),{'backend':self.signature}


def sources(root,count=1,size=(24,18)):
    stage=root/'Stage_3';stage.mkdir()
    for i in range(count):
        im=Image.new('RGB',size,'white');d=ImageDraw.Draw(im)
        d.rectangle((2,2,5,8),fill='red')
        d.rectangle((8,3,11,7),outline='red')  # Separate piece + intentional hole.
        d.point((18,14),fill='red')  # Outside whole-object box: unassigned.
        im.save(stage/f'{100+i} GNR 3.bmp')
    return stage


def review_for(library,rows=None,mask=None,decision='approved'):
    rows=rows or list_objects(library,status='pending_review')
    items=[]
    for row in rows:
        with Image.open(Path(library)/row['folder']/'object.png') as im:
            selected=(np.asarray(im.convert('RGB'))==[255,0,0]).all(axis=2)
        if mask is not None:selected=mask
        items.append(dict(id=row['id'],source_sha256=row['source_sha256'],
            candidate_hash=row['candidate_hash'],category=row['category'],decision=decision,
            mask_runs=_runs(selected)))
    value={'reviewer':'Test designer','reviews':items}
    path=Path(library)/'test_review.json';path.write_text(json.dumps(value))
    return path,value


class SemanticObjectTests(unittest.TestCase):
    def test_whole_disconnected_object_hierarchy_native_pixels_and_omission_map(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);stage=sources(root);library=root/'objects';agent=Vision();sam=Segmenter()
            original=next(stage.iterdir()).read_bytes()
            report=build_object_catalog(stage,library,agent,'fixture-vision',sam,progress=None)
            self.assertEqual(report['objects'],{'pending_review':2})
            self.assertEqual((len(agent.calls),sam.calls),(1,2))
            rows=list_objects(library,status='pending_review')
            animal=next(r for r in rows if r['category']=='peacock')
            feather=next(r for r in rows if r['category']=='feather')
            self.assertEqual(feather['parent_object_id'],animal['id'])
            crop=Image.open(library/animal['folder']/'object.png').convert('RGB')
            expected=Image.open(next(stage.iterdir())).convert('RGB').crop((1,1,14,11))
            self.assertEqual(crop.tobytes(),expected.tobytes())
            self.assertEqual(crop.getpixel((8,3)),(255,255,255))
            self.assertEqual(next(stage.iterdir()).read_bytes(),original)
            audit=json.loads(next((library/'source_audits').glob('*/audit.json')).read_text())
            self.assertEqual(audit['unassigned_red_pixels'],1)
            self.assertFalse(audit['all_objects_found'])
            self.assertEqual(list_objects(library),[])

    def test_budget_resumes_next_source_and_never_repeats_completed_paid_calls(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);stage=sources(root,2);library=root/'objects';agent=Vision();sam=Segmenter()
            for i in (1,2):
                report=build_object_catalog(stage,library,agent,'fixture-vision',sam,
                    max_new_calls=1,progress=None)
                self.assertEqual(report['new_api_calls'],1)
                self.assertEqual(len(agent.calls),i)
            report=build_object_catalog(stage,library,agent,'fixture-vision',sam,progress=None)
            self.assertEqual(report['new_api_calls'],0)
            self.assertEqual(len(agent.calls),2)
            changed=next(stage.iterdir());Image.new('RGB',(24,18),'white').save(changed)
            with self.assertRaisesRegex(ValueError,'changed/was removed'):
                build_object_catalog(stage,library,agent,'fixture-vision',sam,progress=None)

    def test_overlapping_scan_partial_source_resume_and_bad_proposals(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);stage=sources(root,size=(64,50));library=root/'objects';agent=Vision();sam=Segmenter()
            first=build_object_catalog(stage,library,agent,'fixture-vision',sam,
                detail_scan=True,window=32,overlap=8,max_new_calls=1,progress=None)
            self.assertEqual(first['sources'],{'processing':1})
            second=build_object_catalog(stage,library,agent,'fixture-vision',sam,
                detail_scan=True,window=32,overlap=8,max_new_calls=20,progress=None)
            self.assertEqual(second['sources'],{'catalogued':1})
            self.assertEqual(len(agent.calls),len(views((64,50),True,32,8)))
            self.assertEqual(second['new_api_calls'],len(agent.calls)-1)
            valid,bad=validate_proposals({'objects':[
                {'id':'o1','category':'leaf','bbox':[-1,0,8,8]},
                {'id':'o2','category':'leaf','bbox':[1.,1,8,8]},
                {'id':'o3','category':'leaf','bbox':[1,1,8,8]}]},(24,18))
            self.assertEqual((len(valid),len(bad)),(1,2))

    def test_review_approval_deduplication_source_instances_and_clean_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);stage=sources(root,2);library=root/'objects'
            build_object_catalog(stage,library,Vision(), 'fixture-vision',Segmenter(),progress=None)
            path,value=review_for(library)
            report=import_reviews(library,path)
            self.assertEqual(report['imported_reviews'],4)
            approved=list_objects(library,category='peacock',width=10,height=7)
            self.assertEqual(len(approved),2)
            self.assertEqual(len({a['asset_id'] for a in approved}),1)
            self.assertEqual(approved[0]['export_bbox'],[2,2,10,7])
            self.assertEqual(import_reviews(library,path)['imported_reviews'],0)
            out=root/'export';report=export_current_approved(library,out)
            self.assertEqual(report['source_instances'],4)
            self.assertEqual(len(list(out.rglob('*.png'))),report['unique_assets'])
            self.assertEqual(report['unique_assets'],2)
            for a in approved:
                png=Image.open(library/a['approved_png']).convert('RGB')
                bmp=Image.open(library/a['approved_bmp']).convert('RGB')
                self.assertEqual(png.tobytes(),bmp.tobytes())
                self.assertFalse(a['business_rules_verified'])
            asset=library/approved[0]['approved_png'];original_asset=asset.read_bytes()
            Image.new('RGB',(10,7),'red').save(asset)
            with self.assertRaisesRegex(ValueError,'approved object changed'):
                export_current_approved(library,root/'corrupt_export')
            self.assertFalse((root/'corrupt_export').exists())
            asset.write_bytes(original_asset)
            with self.assertRaisesRegex(ValueError,'new export folder'):
                export_current_approved(library,out)
            # Rejecting an old approval must remove it from current exports.
            for item in value['reviews']:item['decision']='rejected';item['mask_runs']=[]
            path.write_text(json.dumps(value));import_reviews(library,path)
            self.assertEqual(list_objects(library),[])
            with self.assertRaisesRegex(ValueError,'No designer-reviewed'):
                export_current_approved(library,root/'new_export')

    def test_masks_cannot_fill_white_and_invalid_batch_does_not_partially_approve(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);stage=sources(root);library=root/'objects'
            build_object_catalog(stage,library,Vision(),'fixture-vision',Segmenter(),progress=None)
            path,value=review_for(library)
            rows=list_objects(library,status='pending_review')
            value['reviews'][-1]['mask_runs']=[[0,rows[-1]['bbox'][2]*rows[-1]['bbox'][3]]]
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError,'cannot create red'):
                import_reviews(library,path)
            self.assertEqual(list_objects(library),[])
            self.assertEqual(catalog_summary(library)['objects'],{'pending_review':2})
            path,value=review_for(library)
            value['reviews'][0]['mask_runs']=[[0,999999]];path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError,'native bounds'):import_reviews(library,path)

    def test_editor_has_original_context_native_masks_and_escaped_model_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);stage=sources(root);library=root/'objects';agent=Vision()
            agent.plan['objects'][0]['label']='</script><script>dangerous()</script>'
            build_object_catalog(stage,library,agent,'fixture-vision',Segmenter(),progress=None)
            gallery=write_review_gallery(library)
            text=gallery.read_text()
            self.assertEqual(text.count('<script>'),1)
            self.assertEqual(text.count('</script>'),1)
            self.assertIn('canvas',text)
            self.assertIn('mask_runs',text)
            self.assertIn('data:image/png;base64,',text)
            self.assertIn('no public',__import__('textile.object_review',fromlist=['']).__doc__.lower())
            with self.assertRaisesRegex(ValueError,'positive integer'):write_review_gallery(library,limit=-1)

    def test_manual_corrected_box_and_invalid_sources_do_not_send_artwork(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);stage=sources(root);library=root/'objects';agent=Vision();sam=Segmenter()
            Image.new('RGB',(10,10),'pink').save(stage/'invalid.png')
            Image.new('RGBA',(10,10),(255,0,0,128)).save(stage/'transparent.png')
            report=build_object_catalog(stage,library,agent,'fixture-vision',sam,progress=None)
            self.assertEqual(report['sources']['excluded_invalid'],2)
            self.assertEqual(len(agent.calls),1)
            source=stage/'100 GNR 3.bmp'
            key=add_manual_candidate(library,source,'ornament',[17,13,3,3],sam)
            self.assertIn(key,[r['id'] for r in list_objects(library,status='pending_review')])
            self.assertEqual(len(agent.calls),1)
            with self.assertRaisesRegex(ValueError,'does not fit'):
                add_manual_candidate(library,source,'peacock',[20,15,20,20],sam)

    def test_notebook_embeds_current_code_with_empty_outputs(self):
        root=Path(__file__).resolve().parents[1]
        nb=json.loads((root/'notebooks/Stage3_Semantic_Object_Repository.ipynb').read_text())
        payload=None
        for cell in nb['cells']:
            if cell['cell_type']!='code':continue
            source=''.join(cell['source']);ast.parse(source)
            self.assertEqual(cell['outputs'],[])
            self.assertIsNone(cell['execution_count'])
            for node in ast.walk(ast.parse(source)):
                if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PAYLOAD' for t in node.targets):
                    payload=ast.literal_eval(node.value)
        self.assertIsNotNone(payload)
        with zipfile.ZipFile(io.BytesIO(base64.b64decode(payload))) as z:
            for name in ('semantic_objects.py','object_review.py'):
                self.assertEqual(z.read('textile/'+name),(root/'textile'/name).read_bytes())


if __name__=='__main__':unittest.main()
