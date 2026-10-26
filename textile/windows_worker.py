r"""Run validated data-only agent jobs on the Windows Photoshop workstation.

Usage: python -m textile.windows_worker C:\path\to\unpacked_job
No listener, arbitrary script execution, or remote shell is exposed.
"""
import hashlib
import json
import os
from pathlib import Path
import sys
import zipfile
from PIL import Image
from .core import validate
from .photoshop import validate_plan, write_jsx


def package_job(current, plan, destination):
    current=Path(current)
    with Image.open(current) as image:
        size=image.size
        if not validate(image,size)['automated_checks_passed']:
            raise ValueError('Job requires an exact red/white raster.')
    validate_plan(plan,size)
    metadata=dict(width=size[0],height=size[1],sha256=hashlib.sha256(current.read_bytes()).hexdigest(),
                  status='unreviewed_agent_job')
    with zipfile.ZipFile(destination,'w',zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('current.png',current.read_bytes())
        archive.writestr('plan.json',json.dumps(plan,indent=2))
        archive.writestr('job.json',json.dumps(metadata,indent=2))
    return str(destination)


def run_job(folder):
    if os.name!='nt':
        raise RuntimeError('Run this worker on Windows with desktop Photoshop installed.')
    root=Path(folder).resolve()
    current=root/'current.png'
    metadata=json.loads((root/'job.json').read_text())
    if hashlib.sha256(current.read_bytes()).hexdigest()!=metadata['sha256']:
        raise ValueError('Job raster does not match manifest.')
    plan=json.loads((root/'plan.json').read_text())
    with Image.open(current) as image:
        size=image.size
        if size!=(metadata['width'],metadata['height']) or not validate(image,size)['automated_checks_passed']:
            raise ValueError('Job dimensions or palette are invalid.')
    validate_plan(plan,size)
    output=root/'results'
    if output.exists():
        raise ValueError('Results already exist; use a new job folder for each iteration.')
    output.mkdir()
    script=root/'validated_edits.jsx'
    write_jsx(plan,size,script,current)
    js=script.read_text()
    js=js.replace('File.openDialog("Select the exact current PNG reviewed by the agent", "*.png")',
                  'File('+json.dumps(str(current))+')')
    js=js.replace('Folder.selectDialog("Select a NEW output folder for this iteration")',
                  'Folder('+json.dumps(str(output))+')')
    js=js.replace('alert("Agent edits saved. Revalidate the PNG and request the next review. No production approval is implied.");','')
    js=js.replace('alert("Textile agent: " + error.message);','throw error;')
    script.write_text(js)
    try:
        import win32com.client
    except ImportError as exc:
        raise RuntimeError('Install requirements-windows.txt on the Windows PC first.') from exc
    photoshop=win32com.client.Dispatch('Photoshop.Application')
    photoshop.DoJavaScriptFile(str(script))
    exported=output/'agent_corrected.png'
    if not exported.is_file() or not (output/'agent_corrected.psd').is_file():
        raise RuntimeError('Photoshop did not save both expected output files.')
    with Image.open(exported) as image:
        report=validate(image,size)
    report.update(unresolved=plan['unresolved'],status='needs_designer_and_agent_review')
    (output/'worker_report.json').write_text(json.dumps(report,indent=2))
    if not report['automated_checks_passed']:
        raise RuntimeError('Photoshop output failed pixel checks; inspect worker_report.json.')
    return output

if __name__=='__main__':
    if len(sys.argv)!=2:
        raise SystemExit('Usage: python -m textile.windows_worker PATH_TO_UNPACKED_JOB')
    print(run_job(sys.argv[1]))
