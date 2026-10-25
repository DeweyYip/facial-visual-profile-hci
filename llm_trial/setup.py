"""Download pinned official CPU runtime and Qwen GGUFs; Python standard library only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parent
LOCAL = ROOT / 'local'


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def download(url, path, size, digest):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and (size is None or path.stat().st_size == size) and sha256(path) == digest:
        print('Verified existing:', path.name, flush=True)
        return
    temporary = path.with_suffix(path.suffix + '.part')
    print('Downloading:', path.name, '(%0.1f MB)' % (size / 1e6) if size else '(runtime archive)', flush=True)
    try:
        request = urllib.request.Request(url, headers={'User-Agent': 'FVP-HCI-Day31/1'})
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open('wb') as f:
            total = 0
            next_notice = 64 * 1024 * 1024
            for chunk in iter(lambda: response.read(4 * 1024 * 1024), b''):
                f.write(chunk)
                total += len(chunk)
                if total >= next_notice:
                    print('  %0.1f MB downloaded' % (total / 1e6), flush=True)
                    next_notice += 64 * 1024 * 1024
        if (size is not None and temporary.stat().st_size != size) or sha256(temporary) != digest:
            raise ValueError('Download size/SHA256 mismatch: ' + path.name)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def extract(archive, destination):
    destination.mkdir(parents=True, exist_ok=True)
    base = destination.resolve()
    with tarfile.open(archive) as tar:
        for item in tar.getmembers():
            target = (base / item.name).resolve()
            if not target.is_relative_to(base) or not (item.isfile() or item.isdir() or item.issym()):
                raise ValueError('Unsupported archive member: ' + item.name)
            if item.issym() and not (target.parent / item.linkname).resolve().is_relative_to(base):
                raise ValueError('Archive link escapes destination')
        tar.extractall(base)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', choices=('0.6b', '1.7b'), default=['0.6b', '1.7b'])
    args = parser.parse_args()
    if platform.system() != 'Darwin' or platform.machine() != 'x86_64':
        parser.error('This automatic setup is for the confirmed Intel Mac (Darwin x86_64).')
    runtime = json.loads((ROOT / 'runtime_manifest.json').read_text())
    LOCAL.mkdir(parents=True, exist_ok=True)
    (LOCAL / 'runtime_asset.json').write_text(json.dumps(runtime, indent=2))
    archive = LOCAL / runtime['asset_name']
    download(runtime['browser_download_url'], archive, runtime['size'], runtime['sha256'])
    extract(archive, LOCAL / 'runtime')
    candidates = list((LOCAL / 'runtime').rglob('llama-server'))
    if len(candidates) != 1:
        raise ValueError('Expected one llama-server executable in the verified runtime')
    candidates[0].chmod(candidates[0].stat().st_mode | 0o111)
    inventory = {str(p.relative_to(LOCAL / 'runtime')): sha256(p)
                 for p in (LOCAL / 'runtime').rglob('*') if p.is_file()}
    (LOCAL / 'runtime_inventory.json').write_text(json.dumps(inventory, indent=2))
    models = json.loads((ROOT / 'model_manifest.json').read_text())
    for name in args.models:
        spec = models[name]
        url = 'https://huggingface.co/{repo}/resolve/{revision}/{file}'.format(**spec)
        download(url, LOCAL / 'models' / spec['file'], spec['size'], spec['sha256'])
    print('SETUP READY. No app dependencies were changed. Next: python llm_trial/run_trial.py', flush=True)


if __name__ == '__main__':
    main()
