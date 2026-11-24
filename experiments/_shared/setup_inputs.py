"""Stage the pinned source and public inputs on the Colab VM, not in Drive."""
import hashlib
import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys

SPLITS = {
    'eurosat': ('EuroSAT', '1Ip7yaCWFi0eaOFUGga0lUdVi_DDQth1o'),
    'oxford_pets': ('OxfordPets', '1501r8Ber4nNKvmlFVQZ8SeUHTcdTTEqs'),
    'dtd': ('DescribableTextures', '1u3_QfB467jqHgNXC00UIzbLZRQCg2S7x'),
}
CLIP = '40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af'
DINO = 'bf34ad0f424b9029b593e8dc3ed553bf26e88bcba0d32bf3e62a6209cb64c85e'


def fetch(url, path, sha=None):
    if not path.exists():
        tmp = path.with_suffix(path.suffix + '.part')
        subprocess.run(['curl', '-fLsS', '--retry', '3', url, '-o', str(tmp)], check=True)
        tmp.replace(path)
    if sha:
        with path.open('rb') as file:
            assert hashlib.file_digest(file, 'sha256').hexdigest() == sha, path
    return path


def setup(root, data, tags, source=None):
    root, data = Path(root), Path(data)
    if importlib.util.find_spec('pkg_resources') is None:
        subprocess.run([sys.executable, '-m', 'pip', '-q', 'install', 'setuptools<81'], check=True)
    root.mkdir(parents=True, exist_ok=True)
    data.mkdir(parents=True, exist_ok=True)
    if not (root/'upstream').exists():
        shutil.unpack_archive(source or root/'source.zip', root)
    weights = root/'weights'
    weights.mkdir(exist_ok=True)
    fetch(f'https://openaipublic.azureedge.net/clip/models/{CLIP}/ViT-B-32.pt', weights/'ViT-B-32.pt', CLIP)
    fetch('https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth',
          weights/'dino_vitbase16_pretrain.pth', DINO)
    if (root/'upstream') not in map(Path, sys.path):
        sys.path[:0] = [str(root/'upstream'), str(root/'dino-source')]
    for tag in tags:
        if tag == 'cifar10':
            from torchvision.datasets import CIFAR10
            CIFAR10(data/tag, download=True)
            continue
        folder = data/tag
        folder.mkdir(exist_ok=True)
        name, drive_id = SPLITS[tag]
        split = folder/f'split_zhou_{name}.json'
        if not split.exists():
            import gdown
            gdown.download(id=drive_id, output=str(split))
        if tag == 'eurosat' and not (folder/'EuroSAT_RGB').exists():
            archive = fetch('https://zenodo.org/api/records/7711810/files/EuroSAT_RGB.zip/content', folder/'EuroSAT_RGB.zip')
            shutil.unpack_archive(archive, folder)
        if tag == 'oxford_pets' and not (folder/'images').exists():
            archive = fetch('https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz', folder/'images.tar.gz')
            shutil.unpack_archive(archive, folder)
        if tag == 'dtd' and not (folder/'images').exists():
            url = 'https://thor.robots.ox.ac.uk/dtd/dtd-r1.0.1.tar.gz'
            archive = fetch(url, data/'dtd-r1.0.1.tar.gz')
            shutil.unpack_archive(archive, data)
    return root, data
