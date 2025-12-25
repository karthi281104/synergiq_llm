import argparse
import json
import os


def _iter_pdfs(folder: str):
    for root, _dirs, files in os.walk(folder):
        for name in files:
            if name.lower().endswith('.pdf'):
                yield os.path.join(root, name)


def main() -> None:
    parser = argparse.ArgumentParser(description='Create a sources manifest skeleton for mixed (own/open) PDFs.')
    parser.add_argument('--pdfs', default='eval/pdfs', help='Folder containing PDFs to list (recursively)')
    parser.add_argument('--out', default='eval/sources_manifest.json', help='Output manifest JSON path')
    parser.add_argument('--default-source-type', default='own', choices=['own', 'open', 'paid_private'])
    args = parser.parse_args()

    pdfs = sorted(list(_iter_pdfs(args.pdfs)))
    if not pdfs:
        raise SystemExit(f'No PDFs found under: {args.pdfs}')

    payload = {
        'version': 1,
        'defaults': {
            'source_type': args.default_source_type,
            'source_url': '',
            'license': '',
        },
        'pdfs': {},
    }

    for p in pdfs:
        rel = os.path.relpath(p, start=os.getcwd()).replace('\\', '/')
        inferred = args.default_source_type
        # Safe default: anything stored under eval/pdfs/private is treated as non-redistributable
        if '/eval/pdfs/private/' in f'/{rel}':
            inferred = 'paid_private'
        payload['pdfs'][rel] = {
            'source_type': inferred,
            'source_url': '',
            'license': '',
        }

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f'Wrote manifest skeleton: {args.out}')
    print(f'PDFs listed: {len(pdfs)}')


if __name__ == '__main__':
    main()
