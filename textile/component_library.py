"""Central index of exact native red regions from designer Step_3 images.

Region extraction proposes pieces; connected regions are not semantic labels.
Invalid originals are reported and left unchanged. Indexing uses no API calls.
"""
import csv
import hashlib
import json
import sqlite3
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from .components import binary, raster, digest
from .data import EXTENSIONS, pair_key

EMBEDDER = 'clip-ViT-B-32'
SCHEMA_VERSION = 1


def connect(library):
    root=Path(library); root.mkdir(parents=True,exist_ok=True)
    db=sqlite3.connect(root/'library.sqlite3',timeout=60)
    db.execute('CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL)')
    old=db.execute("SELECT value FROM settings WHERE key='schema'").fetchone()
    if old and old[0]!=str(SCHEMA_VERSION):
        db.close(); raise ValueError('Library schema differs; use a new library folder.')
    db.execute("INSERT OR IGNORE INTO settings VALUES ('schema',?)",(str(SCHEMA_VERSION),))
    db.execute('''CREATE TABLE IF NOT EXISTS sources (
        path TEXT PRIMARY KEY, hash TEXT NOT NULL, status TEXT NOT NULL, regions INTEGER NOT NULL, detail TEXT NOT NULL)''')
    db.execute('''CREATE TABLE IF NOT EXISTS pieces (
        id TEXT PRIMARY KEY, source_path TEXT NOT NULL, source_hash TEXT NOT NULL,
        family_id TEXT NOT NULL, width INTEGER NOT NULL, height INTEGER NOT NULL,
        x INTEGER NOT NULL, y INTEGER NOT NULL, red_pixels INTEGER NOT NULL,
        image_hash TEXT NOT NULL, white_hash TEXT NOT NULL,
        embedding BLOB, embedding_model TEXT)''')
    db.execute('CREATE INDEX IF NOT EXISTS native_size ON pieces(width,height)')
    db.execute('CREATE INDEX IF NOT EXISTS raster_hash ON pieces(image_hash)')
    db.commit()
    return db


def piece_paths(library,key):
    import re
    if not re.fullmatch('[0-9a-f]{24}',key or ''): raise ValueError('Invalid component ID.')
    folder=Path(library)/'pieces'/key[:2]/key
    return folder/'piece.png',folder/'protected_white.png'


def load_piece(library,key):
    db=connect(library); db.row_factory=sqlite3.Row
    try: row=db.execute('SELECT * FROM pieces WHERE id=?',(key,)).fetchone()
    finally: db.close()
    if row is None: raise ValueError('Component not found in library.')
    meta={k:row[k] for k in row.keys() if k!='embedding'}
    png,white=piece_paths(library,key)
    im,mask=binary(png)
    with Image.open(white) as opened: protection=opened.convert('L')
    p=np.asarray(protection)==255
    if (digest(im)!=meta['image_hash'] or digest(protection)!=meta['white_hash']
        or im.size!=(meta['width'],meta['height']) or p.shape!=mask.shape or (p&mask).any()):
        raise ValueError('Stored component changed; rebuild from the designer source in a new library.')
    return im,p,meta


def index_designer_images(stage3,library,progress=print):
    root=Path(stage3)
    if not root.is_dir(): raise ValueError(f'Stage_3 folder missing: {root}')
    files=sorted(p.resolve() for p in root.rglob('*') if p.is_file() and p.suffix.lower() in EXTENSIONS)
    if not files: raise ValueError('No designer images found in Stage_3.')
    db=connect(library)
    try:
        # A library is an immutable snapshot: changes/removals require a new folder.
        known=dict(db.execute('SELECT path,hash FROM sources'))
        current={str(p) for p in files}
        if set(known)-current: raise ValueError('Previously indexed source files are absent. Use a new library snapshot.')
        hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
        if any(hashes[p]!=h for p,h in known.items()):
            raise ValueError('A previously indexed designer file changed. Use a new library snapshot.')
        for i,path in enumerate(files,1):
            if str(path) in known: continue
            if progress: progress(f'Extracting designer image {i}/{len(files)}: {path.name}')
            filehash=hashes[str(path)]
            try:
                im,red=binary(path)
            except (ValueError,OSError) as exc:
                detail=f'{type(exc).__name__}: {exc}'
                db.execute('INSERT INTO sources VALUES (?,?,?,?,?)',(str(path),filehash,'excluded_invalid',0,detail))
                db.commit(); continue
            labels,count=ndimage.label(red,np.ones((3,3)))
            slices=ndimage.find_objects(labels)
            for label,box in enumerate(slices,1):
                ys,xs=box
                mask=labels[box]==label
                # Protect original white inside a closed contour, not other red pieces.
                protect=ndimage.binary_fill_holes(mask)&~red[box]
                component=raster(mask); white=Image.fromarray(protect.astype(np.uint8)*255)
                key=hashlib.sha256(f'{path}|{filehash}|{label}|{SCHEMA_VERSION}'.encode()).hexdigest()[:24]
                png,whitepath=piece_paths(library,key); png.parent.mkdir(parents=True,exist_ok=True)
                component.save(png); white.save(whitepath)
                db.execute('INSERT INTO pieces VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)',
                    (key,str(path),filehash,pair_key(path,3),component.width,component.height,xs.start,ys.start,
                     int(mask.sum()),digest(component),digest(white),None,None))
            db.execute('INSERT INTO sources VALUES (?,?,?,?,?)',(str(path),filehash,'indexed',count,
                '8-connected red regions; semantic separation and open internal white channels are not verified.'))
            db.commit()
        source_rows=list(db.execute('SELECT path,status,regions,detail FROM sources ORDER BY path'))
        summary=dict(designer_files=len(files),indexed_files=sum(r[1]=='indexed' for r in source_rows),
            excluded_files=sum(r[1]!='indexed' for r in source_rows),components=db.execute('SELECT COUNT(*) FROM pieces').fetchone()[0],
            extraction='All 8-connected red regions, including single pixels; no minimum-area pruning.',
            semantic_components_verified=False,grouping='Source names retained; cross-design variants are not automatically identified.',
            limitation='Touching semantic pieces can form one region. Disconnected fragments can form several. Extracted candidates are not automatically approved components.')
        Path(library,'index_summary.json').write_text(json.dumps(summary,indent=2))
        with Path(library,'source_audit.csv').open('w',newline='') as handle:
            writer=csv.writer(handle);writer.writerow(['path','status','regions','detail']);writer.writerows(source_rows)
        return summary
    finally: db.close()


def embed_library(library,encoder=None,model=EMBEDDER,progress=print):
    db=connect(library)
    try:
        other=db.execute('SELECT COUNT(*) FROM pieces WHERE embedding IS NOT NULL AND embedding_model!=?',(model,)).fetchone()[0]
        if other: raise ValueError('Embedding model changed; use a new library snapshot.')
        # Repeated identical rasters share an embedding, while all source provenance stays indexed.
        db.execute('''UPDATE pieces SET
            embedding=(SELECT p.embedding FROM pieces p WHERE p.image_hash=pieces.image_hash AND p.embedding_model=? AND p.embedding IS NOT NULL LIMIT 1),
            embedding_model=? WHERE embedding IS NULL AND EXISTS
            (SELECT 1 FROM pieces p WHERE p.image_hash=pieces.image_hash AND p.embedding_model=? AND p.embedding IS NOT NULL)''',(model,model,model))
        db.commit()
        keys=[r[0] for r in db.execute('SELECT MIN(id) FROM pieces WHERE embedding IS NULL GROUP BY image_hash ORDER BY MIN(id)')]
        if not keys: return {'new_embeddings':0,'embedding_model':model}
        if encoder is None:
            from sentence_transformers import SentenceTransformer
            encoder=SentenceTransformer(model)
        for start in range(0,len(keys),32):
            batch=keys[start:start+32]
            images=[load_piece(library,k)[0] for k in batch]
            vectors=np.asarray(encoder.encode(images,batch_size=16,convert_to_numpy=True,show_progress_bar=False),dtype=np.float32)
            if vectors.ndim!=2 or len(vectors)!=len(batch) or not np.isfinite(vectors).all(): raise ValueError('Invalid embeddings.')
            norms=np.linalg.norm(vectors,axis=1,keepdims=True)
            if (norms==0).any(): raise ValueError('Empty embedding vector.')
            vectors=vectors/norms
            db.executemany('UPDATE pieces SET embedding=?,embedding_model=? WHERE image_hash=(SELECT image_hash FROM pieces WHERE id=?)',
                           [(v.tobytes(),model,k) for v,k in zip(vectors,batch)])
            db.commit()
            if progress: progress(f'Indexed similarity for {min(start+32,len(keys))}/{len(keys)} remaining pieces')
        return {'new_embeddings':len(keys),'embedding_model':model}
    finally: db.close()


def search_exact_size(library,width,height,vector,top_k=3,model=EMBEDDER):
    if not all(float(v).is_integer() and v>0 for v in (width,height,top_k)):
        raise ValueError('Dimensions and top_k must be positive integers.')
    query=np.asarray(vector,dtype=np.float32).reshape(-1)
    if not np.isfinite(query).all() or not np.linalg.norm(query): raise ValueError('Invalid query embedding.')
    query=query/np.linalg.norm(query)
    db=connect(library); db.row_factory=sqlite3.Row
    try:
        rows=db.execute('SELECT id,image_hash,white_hash,embedding,embedding_model FROM pieces WHERE width=? AND height=? ORDER BY id',(int(width),int(height))).fetchall()
    finally: db.close()
    ranked=[]; seen=set()
    for row in rows:
        if row['embedding'] is None or row['embedding_model']!=model:
            raise ValueError('Complete the library embedding step before searching.')
        signature=(row['image_hash'],row['white_hash'])
        if signature in seen: continue
        seen.add(signature)
        v=np.frombuffer(row['embedding'],dtype=np.float32)
        if v.shape!=query.shape: raise ValueError('Query and library embedding dimensions differ.')
        ranked.append({'id':row['id'],'similarity':float(query@v)})
    return sorted(ranked,key=lambda r:(-r['similarity'],r['id']))[:int(top_k)]
