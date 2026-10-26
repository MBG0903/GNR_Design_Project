import base64
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from PIL import Image
from textile.genai import generate, review
from textile.photoshop import validate_plan, write_jsx
from textile.windows_worker import package_job

class AgentTests(unittest.TestCase):
    def plan(self):
        return dict(findings=[],unresolved=['Inspect taper'],operations=[dict(rule='R2',reason='Separate pieces',colour='white',points=[[1,1],[3,1],[3,3],[1,3]])])

    def test_plan_rejects_code_out_of_bounds_and_large_edits(self):
        validate_plan(self.plan(),(100,100))
        for points in [[[0,0],[100,0],[100,100]], [[-1,0],[1,0],[1,1]], [['evil',0],[1,0],[1,1]]]:
            plan=self.plan();plan['operations'][0]['points']=points
            with self.assertRaises(ValueError):validate_plan(plan,(100,100))
        plan=self.plan();plan['operations'][0]['command']='app.quit()'
        with self.assertRaises(ValueError):validate_plan(plan,(100,100))

    def test_generation_restores_native_size_and_palette_from_mock_api(self):
        image=Image.new('RGB',(1024,1024),'white');image.paste('red',(100,100,500,500))
        buf=io.BytesIO();image.save(buf,format='PNG')
        calls=[]
        def edit(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(data=[SimpleNamespace(b64_json=base64.b64encode(buf.getvalue()).decode())])
        client=SimpleNamespace(images=SimpleNamespace(edit=edit))
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'source.png';Image.new('RGB',(401,340),'white').save(source)
            draft=generate(source,Path(tmp)/'out',client=client)
            with Image.open(draft) as result:
                self.assertEqual(result.size,(401,340))
                self.assertLessEqual(set(result.getdata()),{(255,0,0),(255,255,255)})
            self.assertIn('Image 1',calls[0]['prompt'])

    def test_agent_review_and_photoshop_job_packaging(self):
        plan=self.plan()
        client=SimpleNamespace(responses=SimpleNamespace(create=lambda **kwargs:SimpleNamespace(output_text=json.dumps(plan))))
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source.png';current=root/'current.png'
            im=Image.new('RGB',(100,100),'white');im.putpixel((50,50),(255,0,0));im.save(current);im.save(source)
            actual=review(source,current,root/'plan.json',client=client,model='mock-astra')
            self.assertEqual(actual,plan)
            script=Path(write_jsx(plan,(100,100),root/'edit.jsx',current)).read_text()
            self.assertIn('SelectionType.REPLACE, 0, false',script)
            self.assertIn('checksum',script)
            self.assertNotIn('__OPS__',script)
            bundle=package_job(current,plan,root/'job.zip')
            with zipfile.ZipFile(bundle) as archive:
                self.assertEqual(set(archive.namelist()),{'current.png','plan.json','job.json'})
