#!/usr/bin/env python3
"""Fetch the pinned third-party sources for the evaluator; do not commit them."""
import io
from pathlib import Path
import subprocess
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    'upstream': ('https://github.com/fazliimam/NoLA.git', 'caec2722bba84bd4807256645c5cb75d96d7ed12'),
    'dino-source': ('https://github.com/facebookresearch/dino.git', '7c446df5b9f45747937fb0d72314eb9f7b66930a'),
}


def git(*args):
    return subprocess.check_output(['git', *map(str, args)])


def main():
    for name, (url, revision) in SOURCES.items():
        folder = ROOT / name
        if not folder.exists():
            git('init', '-q', folder)
        try:
            git('-C', folder, 'cat-file', '-e', f'{revision}^{{commit}}')
        except subprocess.CalledProcessError:
            git('-C', folder, 'fetch', '--quiet', '--depth=1', url, revision)
        if not (folder / 'README.md').exists():
            git('-C', folder, 'checkout', '--quiet', '--detach', revision)
        if git('-C', folder, 'rev-parse', 'HEAD').decode().strip() != revision:
            raise SystemExit(f'{folder} is not at the pinned revision; use a clean clone')
    output = ROOT / 'final-evaluation' / 'source.zip'
    tmp = output.with_suffix('.tmp')
    with ZipFile(tmp, 'w', ZIP_DEFLATED) as bundle:
        for name, (_, revision) in SOURCES.items():
            archive = git('-C', ROOT / name, 'archive', '--format=zip', f'--prefix={name}/', revision)
            with ZipFile(io.BytesIO(archive)) as source:
                for item in source.infolist():
                    if not item.is_dir():
                        bundle.writestr(item.filename, source.read(item))
    tmp.replace(output)
    print(output)


if __name__ == '__main__':
    main()
