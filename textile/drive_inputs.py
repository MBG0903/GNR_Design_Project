"""Stage known artwork assets on mounted Drive without replacing changed files."""
import hashlib
from pathlib import Path
from urllib.request import urlopen


def stage_files(destination, assets, uploads=None, fetch=None):
    """Reuse matching Drive files, otherwise copy uploads or download pinned files.

    Hash checks apply before saving; only assets supplied by the caller are read.
    Mounting Drive is the notebook's responsibility. No credentials are involved.
    """
    destination=Path(destination);uploads=Path(uploads) if uploads is not None else None
    planned=[];report=[]
    for asset in assets:
        relative=Path(asset['path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Asset paths must remain inside the dataset folder.')
        target=destination/relative
        if target.exists():
            if hashlib.sha256(target.read_bytes()).hexdigest()!=asset['sha256']:
                raise ValueError(f'{relative}: existing Drive file differs; it was not overwritten.')
            report.append(dict(path=str(target),origin='existing_drive_file'));continue
        uploaded=None
        if uploads is not None and asset.get('upload_name'):
            name=asset['upload_name']
            if Path(name).name!=name:raise ValueError('Expected a plain uploaded filename.')
            candidate=uploads/name
            if candidate.is_file():uploaded=candidate
        planned.append((asset,target,uploaded))
    # Existing-file conflicts are detected before any new asset is copied.
    for asset,target,uploaded in planned:
        if uploaded is not None:
            content=uploaded.read_bytes();origin='colab_upload'
        else:
            if fetch is not None:content=fetch(asset['url'])
            else:
                with urlopen(asset['url'],timeout=30) as response:content=response.read()
            origin='pinned_github_download'
        if hashlib.sha256(content).hexdigest()!=asset['sha256']:
            raise ValueError(f'{asset["path"]}: content hash differs; the file was not saved.')
        target.parent.mkdir(parents=True,exist_ok=True)
        # Exclusive creation prevents silently replacing an input written concurrently.
        with target.open('xb') as stream:stream.write(content)
        report.append(dict(path=str(target),origin=origin))
    return report
