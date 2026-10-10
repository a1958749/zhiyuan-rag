"""Freeze dev-selected settings, source/data identities and first30 paid holdout IDs."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'devsupport/src'))


def main():
    """Preregister without model inference, credential access or reading test outputs."""
    from devsupport.evaluate import build_contract, freeze
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    freeze(args.output, build_contract(args.config, ROOT))
    print('Frozen evaluation contract:', args.output)


if __name__ == '__main__':
    main()
