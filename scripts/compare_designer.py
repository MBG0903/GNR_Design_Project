"""Reproducible comparison with native-coordinate metrics and diagnostics."""
import argparse,json,hashlib
from pathlib import Path
from PIL import Image,ImageDraw
from textile.comparison import compare

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('designer');parser.add_argument('candidate');parser.add_argument('output')
    args=parser.parse_args();output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    with Image.open(args.designer) as im:designer=im.convert('RGB')
    with Image.open(args.candidate) as im:candidate=im.convert('RGB')
    report,diff=compare(designer,candidate)
    report.update(designer_file_sha256=hashlib.sha256(Path(args.designer).read_bytes()).hexdigest(),
                  candidate_file_sha256=hashlib.sha256(Path(args.candidate).read_bytes()).hexdigest(),
                  evaluation_caveat='The same-design approved Step 3 was previously used as density guidance for the generated draft. This is reference-guided reconstruction similarity, not independent validation or proof of generalization.')
    diff.save(output/'difference_native.png')
    scale=3;w,h=designer.size
    panel=Image.new('RGB',(3*w*scale,h*scale+55),'white');draw=ImageDraw.Draw(panel)
    for i,(title,frame) in enumerate([('Designer Step 3',designer),('Latest generated + corrected',candidate),('Difference',diff)]):
        panel.paste(frame.resize((w*scale,h*scale),Image.Resampling.NEAREST),(i*w*scale,30))
        draw.text((i*w*scale+8,8),title,fill='black')
    draw.text((8,h*scale+36),'Red: both red | White: both white | Blue: missing designer red | Orange: extra red',fill='black')
    panel.save(output/'designer_candidate_difference.png')
    (output/'metrics.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
