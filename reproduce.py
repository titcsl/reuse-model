import gzip
import hashlib
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from analyze import DATASETS


ROOT = Path(__file__).resolve().parent


def file_hash(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def prepare_data():
    for filename, expected_hash in DATASETS.values():
        destination = ROOT / 'data' / filename
        if destination.exists():
            if file_hash(destination) != expected_hash:
                raise ValueError(f'Input checksum mismatch: {filename}')
            continue
        archive = destination.with_suffix('.csv.gz')
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as output:
                temporary_path = Path(output.name)
                with gzip.open(archive, 'rb') as source:
                    shutil.copyfileobj(source, output)
            if file_hash(temporary_path) != expected_hash:
                raise ValueError(f'Decompressed input checksum mismatch: {archive.name}')
            temporary_path.replace(destination)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()


def main():
    prepare_data()
    for filename in ['analyze.py', 'render_results.py', 'verify_artifact.py']:
        subprocess.run([sys.executable, str(ROOT / filename)], cwd=ROOT, check=True)
    print('Reproduction complete. Results are in results/ and figures/.')


if __name__ == '__main__':
    main()
