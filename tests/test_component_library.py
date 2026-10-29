import ast
import base64
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw

from textile.components import place_component, raster
from textile.component_library import (index_designer_images, connect, load_piece,
                                       embed_library, search_exact_size)
from textile.component_agent import run_component_agent, validate_plan


class Encoder:
    def encode(self, images, **kwargs):
        return np.asarray([[1, im.width, im.height] for im in images],dtype=np.float32)


class Agent:
    def __init__(self,plan):
        self.plan=plan;self.calls=[];self.responses=self
    def create(self,**kwargs):
        self.calls.append(kwargs)
        prompt=kwargs['input'][0]['content'][0]['text']
        if 'candidate ID' in prompt:
            import re
            key=re.search('candidate ID ([a-f0-9]+)',prompt).group(1)
            value={'candidate_id':key,'reason':'Fixture exact shape'}
        else: value=self.plan
        return SimpleNamespace(output_text=json.dumps(value))


def sources(root):
    stage=root/'Stage_3';stage.mkdir()
    im=Image.new('RGB',(30,20),'white');d=ImageDraw.Draw(im)
    d.rectangle((2,2,4,5),fill='red')  # 3×4
    d.rectangle((12,2,14,5),fill='red') # identical component, separate provenance
    d.rectangle((20,10,24,14),outline='red') # 5×5 ring
    im.save(stage/'100 GNR 3.bmp')
    Image.new('RGB',(10,10),'pink').save(stage/'101 GNR 3.bmp')
    return stage


class ComponentTests(unittest.TestCase):
    def test_extract_every_region_preserve_pixels_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);stage=sources(root);library=root/'library'
            originals={p:p.read_bytes() for p in stage.iterdir()}
            report=index_designer_images(stage,library,progress=None)
            self.assertEqual((report['indexed_files'],report['excluded_files'],report['components']),(1,1,3))
            db=connect(library)
            rows=db.execute('SELECT id,x,y,width,height FROM pieces').fetchall();db.close()
            for key,x,y,w,h in rows:
                piece,protection,meta=load_piece(library,key)
                expected=Image.open(stage/'100 GNR 3.bmp').convert('RGB').crop((x,y,x+w,y+h))
                self.assertEqual(piece.tobytes(),expected.tobytes())
                if w==5:self.assertEqual(protection.sum(),9)
            self.assertEqual(index_designer_images(stage,library,progress=None)['components'],3)
            for p,content in originals.items():self.assertEqual(p.read_bytes(),content)
            Image.new('RGB',(30,20),'white').save(stage/'100 GNR 3.bmp')
            with self.assertRaisesRegex(ValueError,'changed'):
                index_designer_images(stage,library,progress=None)

    def test_size_filter_embedding_deduplication_and_tamper_detection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);library=root/'library'
            index_designer_images(sources(root),library,progress=None)
            self.assertEqual(embed_library(library,Encoder(),progress=None)['new_embeddings'],2)
            self.assertEqual(embed_library(library,Encoder(),progress=None)['new_embeddings'],0)
            hits=search_exact_size(library,3,4,[1,3,4])
            self.assertEqual(len(hits),1) # identical shapes not repeated in shortlist
            self.assertEqual(search_exact_size(library,4,3,[1,4,3]),[]) # no rotation
            self.assertEqual(search_exact_size(library,4,4,[1,4,4]),[]) # no resizing
            from textile.component_library import piece_paths
            png,_=piece_paths(library,hits[0]['id'])
            Image.new('RGB',(3,4),'white').save(png)
            with self.assertRaisesRegex(ValueError,'changed'):load_piece(library,hits[0]['id'])

    def test_clearance_protected_white_bounds_and_exact_translation(self):
        base=np.zeros((12,20),bool);base[3:7,1:4]=True
        component=raster(np.ones((4,3),bool));protection=np.zeros((4,3),bool)
        out,protected,report=place_component(raster(base),component,protection,6,3)
        self.assertEqual(report['minimum_white_gap_to_existing'],2)
        expected=base.copy();expected[3:7,6:9]=True
        self.assertEqual(out.tobytes(),raster(expected).tobytes())
        with self.assertRaisesRegex(ValueError,'only 1'):
            place_component(raster(base),component,protection,5,3)
        with self.assertRaisesRegex(ValueError,'overlaps'):
            place_component(raster(base),component,protection,3,3)
        with self.assertRaisesRegex(ValueError,'fit'):
            place_component(raster(base),component,protection,19,3)
        p=np.zeros_like(base);p[3:7,6:9]=True
        with self.assertRaisesRegex(ValueError,'protected'):
            place_component(raster(base),component,protection,6,3,p)
        # A large enclosed hole stays white even when a dot is far from its red boundary.
        ring=np.zeros((13,13),bool);ring[0,:]=ring[-1,:]=True;ring[:,0]=ring[:,-1]=True
        p=~ring
        blank=Image.new('RGB',(20,20),'white')
        out,protected,_=place_component(blank,raster(ring),p,1,1)
        with self.assertRaisesRegex(ValueError,'protected'):
            place_component(out,raster(np.ones((1,1),bool)),np.zeros((1,1),bool),7,7,protected)

    def test_agent_exact_paste_unmatched_collision_partial_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);library=root/'library'
            index_designer_images(sources(root),library,progress=None)
            embed_library(library,Encoder(),progress=None)
            plan={'pieces':[
                {'family':'leaf','bbox':[1,1,3,4],'uncertain':False},
                {'family':'leaf','bbox':[5,1,3,4],'uncertain':False}, # one white pixel gap -> blocked
                {'family':'petal','bbox':[10,1,7,7],'uncertain':False}, # no dimension match
                {'family':'stem','bbox':[1,10,3,4],'uncertain':True},
                {'family':'leaf','bbox':[10,10,3,4],'uncertain':False}, # deferred by limit
            ],'unresolved':['fixture uncertainty']}
            client=Agent(plan);sketch=Image.new('RGB',(25,20),'white')
            report=run_component_agent(sketch,library,root/'run',client,'mock-vision',Encoder(),max_pieces=4,progress=None)
            self.assertEqual(report['placed'],1)
            self.assertEqual([r['status'] for r in report['records']],
                ['placed_for_review','placement_blocked','no_exact_size_match','needs_semantic_review','deferred_by_run_limit'])
            self.assertEqual(len(client.calls),3) # planner + two eligible choices
            out=Image.open(root/'run/step_3_review.png').convert('RGB')
            expected=np.zeros((20,25),bool);expected[1:5,1:4]=True
            self.assertEqual(out.tobytes(),raster(expected).tobytes())
            self.assertEqual(out.tobytes(),Image.open(root/'run/step_3_review.bmp').convert('RGB').tobytes())
            self.assertFalse(report['production_ready'])
            run_component_agent(sketch,library,root/'run',client,'mock-vision',Encoder(),max_pieces=4,progress=None)
            self.assertEqual(len(client.calls),3)
            with self.assertRaisesRegex(ValueError,'changed'):
                run_component_agent(sketch,library,root/'run',client,'different-model',Encoder(),max_pieces=4)

    def test_untrusted_agent_coordinates_and_interruption(self):
        for box in ([0,0,11,2],[-1,0,2,2],[0,0,1.5,2],[True,0,2,2]):
            with self.assertRaises(ValueError):
                validate_plan({'pieces':[{'family':'leaf','bbox':box,'uncertain':False}]},(10,10))
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);library=root/'library'
            index_designer_images(sources(root),library,progress=None)
            embed_library(library,Encoder(),progress=None)
            class Fail:
                responses=None
                def __init__(self):self.responses=self;self.calls=0
                def create(self,**kwargs):self.calls+=1;raise RuntimeError('secret error')
            client=Fail();sketch=Image.new('RGB',(10,10),'white')
            with self.assertRaises(RuntimeError):run_component_agent(sketch,library,root/'run',client,'mock',Encoder())
            with self.assertRaisesRegex(ValueError,'Interrupted'):
                run_component_agent(sketch,library,root/'run',client,'mock',Encoder())
            self.assertEqual(client.calls,1)
            self.assertNotIn('secret error',(root/'run/agent_plan/failure.json').read_text())

    def test_notebook_compiles_and_runtime_matches(self):
        root=Path(__file__).resolve().parents[1]
        notebook=json.loads((root/'notebooks/Central_Component_Repository.ipynb').read_text())
        found=False
        for i,cell in enumerate(notebook['cells']):
            if cell['cell_type']!='code':continue
            source=''.join(cell['source']);compile(source,f'cell{i}','exec')
            for node in ast.parse(source).body:
                if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PAYLOAD' for t in node.targets):
                    found=True
                    with zipfile.ZipFile(io.BytesIO(base64.b64decode(ast.literal_eval(node.value)))) as z:
                        for name in z.namelist():
                            self.assertTrue(name.endswith('.py'))
                            self.assertEqual(z.read(name),(root/name).read_bytes())
        self.assertTrue(found)
