"""Offline native-pixel mask review; no public web server or artwork upload."""
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .genai import data_url
from .components import digest
from .semantic_objects import CATEGORIES, list_objects


def _runs(mask):
    a=np.asarray(mask,bool).ravel();d=np.diff(np.r_[False,a,False].astype(np.int8))
    return [[int(s),int(e-s)] for s,e in zip(np.flatnonzero(d==1),np.flatnonzero(d==-1))]


def write_review_gallery(library, output=None, limit=30):
    if type(limit) is not int or limit<1:raise ValueError('Gallery limit must be a positive integer.')
    root=Path(library);records=list_objects(root,status='pending_review')[:limit]
    if not records:raise ValueError('No pending candidates. Run discovery or add a manual box first.')
    items=[];sources={}
    for row in records:
        folder=root/row['folder']
        with Image.open(folder/'source_crop.png') as opened:crop=opened.convert('RGB')
        with Image.open(folder/'object.png') as opened:piece=opened.convert('RGB')
        if digest(crop)!=row['crop_hash'] or digest(piece)!=row['candidate_hash']:
            raise ValueError('A candidate crop changed. Preserve it and use a new catalogue snapshot.')
        with Image.open(folder/'selection_mask.png') as opened:mask=np.asarray(opened.convert('L'))==255
        if mask.shape!=(crop.height,crop.width) or not np.array_equal(mask,(np.asarray(piece)==[255,0,0]).all(axis=2)):
            raise ValueError('Candidate selection mask and object pixels disagree.')
        items.append({k:row[k] for k in ('id','source_id','source_sha256','candidate_hash','category','label','bbox','notes','parent_object_id')})
        items[-1].update(image=data_url(crop),mask_runs=_runs(mask))
        if row['source_id'] not in sources:
            with Image.open(row['source_path']) as opened:source=opened.convert('RGB')
            source.thumbnail((900,900),Image.Resampling.NEAREST)  # Context display only.
            sources[row['source_id']]=dict(image=data_url(source),name=Path(row['source_path']).name)
    payload=json.dumps({'items':items,'sources':sources,'categories':CATEGORIES}).replace('<','\\u003c').replace('&','\\u0026')
    output=Path(output or root/'review_gallery.html')
    output.write_text(TEMPLATE.replace('__PAYLOAD__',payload))
    return output


TEMPLATE=r'''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>GNR object isolation review</title>
<style>
body{font:16px/1.45 system-ui,sans-serif;background:#f4f5f6;color:#17212b;margin:20px}
header{position:sticky;top:0;background:#fff;padding:12px;z-index:2;border:1px solid #ccc}
article{background:white;border:1px solid #ccc;border-radius:6px;margin:24px 0;padding:16px}
canvas{image-rendering:pixelated;touch-action:none;border:1px solid #999;max-width:none}
.panels{display:flex;gap:15px;overflow:auto;max-height:650px}.panel{flex-shrink:0}
button,select,input{font:inherit;margin:4px;padding:5px}details img{max-width:100%;image-rendering:pixelated}
.hint{font-size:14px;color:#485460}h2{font-size:19px}textarea{width:90%;height:45px}
</style><header><strong>GNR whole-object isolation review</strong><br>
Reviewer <input id="reviewer" aria-label="Reviewer name" placeholder="Designer name">
<button id="save">Download reviewed masks JSON</button><span id="count"></span>
<p class="hint">Each card is one semantic object and may contain disconnected parts. Original pixels are on the left; proposed isolated pixels on the right. Paint on the right to include/exclude original red pixels. No new red is drawn, and original white cannot be filled. Native source pixels stay unchanged. Approve isolation only after checking the complete object and excluding unrelated neighbours. Staircase/business-rule approval is separate.</p></header>
<main id="cards"></main><script>
'use strict';const DATA=__PAYLOAD__;const states=[];
function el(tag,text,parent){const e=document.createElement(tag);if(text!==null)e.textContent=text;if(parent)parent.append(e);return e;}
function button(text,parent,fn){const b=el('button',text,parent);b.onclick=fn;return b;}
function rle(a){const out=[];for(let i=0;i<a.length;){if(!a[i]){i++;continue;}const s=i;while(i<a.length&&a[i])i++;out.push([s,i-s]);}return out;}
for(const item of DATA.items){
 const [x,y,w,h]=item.bbox,keep=new Uint8Array(w*h);for(const [s,n] of item.mask_runs)keep.fill(1,s,s+n);
 const st={item,keep,decision:'pending',category:item.category,notes:'',ready:false};states.push(st);
 const card=el('article',null,document.getElementById('cards'));
 el('h2',`${item.category}: ${item.label}`,card);
 el('p',`${DATA.sources[item.source_id].name} | x=${x}, y=${y}, width=${w}, height=${h} | ${item.id}`,card).className='hint';
 if(item.notes)el('p',item.notes,card).className='hint';
 if(item.parent_object_id)el('p',`Parent object: ${item.parent_object_id}`,card).className='hint';
 const details=el('details',null,card);el('summary','Show full source context',details);
 const context=el('img',null,details);context.src=DATA.sources[item.source_id].image;context.alt='Full original textile context';
 const tools=el('div',null,card);let zoom=4,brush=1,mode='include';
 const z=el('select',null,tools);for(const v of [1,2,4,8,12]){const o=el('option',`${v}× native pixels`,z);o.value=v;}z.value=4;
 const sz=el('select',null,tools);for(const v of [1,2,4,8]){const o=el('option',`${v}-pixel brush`,sz);o.value=v;}sz.value=1;sz.onchange=()=>brush=+sz.value;
 button('Include red',tools,()=>mode='include');button('Exclude red',tools,()=>mode='exclude');
 const panels=el('div',null,card);panels.className='panels';const p1=el('div',null,panels),p2=el('div',null,panels);p1.className=p2.className='panel';
 el('p','Original source crop',p1);el('p','Isolated object — paint here',p2);
 const a=el('canvas',null,p1),b=el('canvas',null,p2);a.width=b.width=w;a.height=b.height=h;
 const ac=a.getContext('2d'),bc=b.getContext('2d');let original=null,red=null;
 const resize=()=>{for(const c of [a,b]){c.style.width=`${w*zoom}px`;c.style.height=`${h*zoom}px`;}};
 z.onchange=()=>{zoom=+z.value;resize();};resize();
 function render(){if(!original)return;const image=bc.createImageData(w,h);for(let i=0;i<w*h;i++){const show=keep[i]&&red[i];image.data[i*4]=255;image.data[i*4+1]=show?0:255;image.data[i*4+2]=show?0:255;image.data[i*4+3]=255;}bc.putImageData(image,0,0);}
 const im=new Image();im.onload=()=>{ac.drawImage(im,0,0);original=ac.getImageData(0,0,w,h);red=new Uint8Array(w*h);for(let i=0;i<w*h;i++)red[i]=original.data[4*i]===255&&original.data[4*i+1]===0&&original.data[4*i+2]===0;st.ready=true;render();};im.src=item.image;
 let dragging=false,last=null;
 function paintAt(xx,yy){for(let oy=0;oy<brush;oy++)for(let ox=0;ox<brush;ox++){const px=xx+ox,py=yy+oy;if(px>=0&&py>=0&&px<w&&py<h&&red[py*w+px])keep[py*w+px]=mode==='include'?1:0;}}
 function paint(e){if(!red)return;const r=b.getBoundingClientRect(),xx=Math.floor((e.clientX-r.left-b.clientLeft)*w/b.clientWidth),yy=Math.floor((e.clientY-r.top-b.clientTop)*h/b.clientHeight);if(last){const dx=xx-last[0],dy=yy-last[1],n=Math.max(1,Math.abs(dx),Math.abs(dy));for(let i=0;i<=n;i++)paintAt(Math.round(last[0]+dx*i/n),Math.round(last[1]+dy*i/n));}else paintAt(xx,yy);last=[xx,yy];render();}
 b.onpointerdown=e=>{dragging=true;last=null;b.setPointerCapture(e.pointerId);paint(e);};b.onpointermove=e=>{if(dragging)paint(e);};b.onpointerup=()=>{dragging=false;last=null;};b.onpointercancel=()=>{dragging=false;last=null;};
 button('Reset proposed mask',tools,()=>{keep.fill(0);for(const [s,n] of item.mask_runs)keep.fill(1,s,s+n);render();});
 button('Include all original red in box',tools,()=>{if(red)keep.set(red);render();});
 el('p','If an object extends beyond this box, reject it and add a larger box in Colab. This panel cannot invent or recover out-of-box pixels.',card).className='hint';
 const row=el('div',null,card),cat=el('select',null,row);for(const c of DATA.categories){const o=el('option',c,cat);o.value=c;}cat.value=item.category;cat.onchange=()=>st.category=cat.value;
 const decision=el('select',null,row);for(const [v,label] of [['pending','Not reviewed'],['approved','Approve complete object isolation'],['rejected','Reject / needs another box']]){const o=el('option',label,decision);o.value=v;}decision.onchange=()=>{st.decision=decision.value;update();};
 const note=el('textarea',null,card);note.placeholder='Designer notes';note.oninput=()=>st.notes=note.value;
}
function update(){document.getElementById('count').textContent=` ${states.filter(s=>s.decision!=='pending').length}/${states.length} reviewed`;}
document.getElementById('save').onclick=()=>{
 const reviewer=document.getElementById('reviewer').value.trim();if(!reviewer){alert('Enter the designer/reviewer name.');return;}
 const selected=states.filter(s=>s.decision!=='pending');if(!selected.length){alert('Review at least one object first.');return;}
 if(selected.some(s=>!s.ready)){alert('Wait for the source crops to load.');return;}
 const reviews=selected.map(s=>({id:s.item.id,source_sha256:s.item.source_sha256,candidate_hash:s.item.candidate_hash,decision:s.decision,category:s.category,notes:s.notes,mask_runs:s.decision==='approved'?rle(s.keep):[]}));
 if(reviews.some(r=>r.decision==='approved'&&!r.mask_runs.length)){alert('An approved object must contain source red pixels.');return;}
 const blob=new Blob([JSON.stringify({reviewer,reviews},null,2)],{type:'application/json'});const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download='gnr_object_reviews.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
};update();
</script></html>'''
