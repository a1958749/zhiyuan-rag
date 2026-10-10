"""Start with Linux models Python: devsupport/scripts/serve_gpu.py --unload-before-rerank."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))


def main() -> None:
    """Bind only loopback, serializing inference inside the app."""
    import uvicorn
    from devsupport.model_service import create_app

    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8093)
    parser.add_argument('--unload-before-rerank', action='store_true')
    args = parser.parse_args()
    uvicorn.run(create_app(args.unload_before_rerank), host='127.0.0.1', port=args.port,
                workers=1, access_log=False)


if __name__ == '__main__':
    main()
