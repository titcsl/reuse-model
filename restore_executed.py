import argparse
import base64
import csv
import hashlib
from pathlib import Path
import zlib


def recover(csv_path):
    values = {}
    with csv_path.open(newline='', encoding='utf-8-sig') as handle:
        for row in csv.DictReader(handle):
            if row['record_type'] == 'metadata' and row['session'] == '':
                values[row['metric']] = row['value']
            if {'payload_source_zlib_base64', 'payload_sha256'} <= values.keys():
                break
    source = zlib.decompress(base64.b64decode(values['payload_source_zlib_base64'], validate=True))
    actual = hashlib.sha256(source).hexdigest()
    if actual != values['payload_sha256']:
        raise ValueError('The embedded source does not match its recorded checksum')
    return source, actual


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('csv', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    source, checksum = recover(args.csv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('xb') as handle:
        handle.write(source)
    print(f'Restored {args.output}\nSHA-256: {checksum}')


if __name__ == '__main__':
    main()
