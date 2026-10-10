"""Start the original DocsGPT ASGI app using a process-only DPAPI credential."""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def configure() -> None:
    """Load the configured secret into this process before importing native settings."""
    from devsupport.src.devsupport.credentials import load_credential
    os.environ['DOCSGPT_ENV_FILE'] = str(ROOT / '.env')
    os.environ['DOCSGPT_HOME'] = str(ROOT / 'devsupport/runtime/docsgpt')
    os.environ['MODELS_CONFIG_DIR'] = str(ROOT / 'devsupport/configs/models')
    os.environ['LLM_PROVIDER'] = 'openai_compatible'
    os.environ['LLM_NAME'] = 'devsupport-opus'
    # An inherited mock endpoint otherwise overrides the catalog with the same ID.
    os.environ['OPENAI_BASE_URL'] = ''
    os.environ['EMBEDDINGS_BASE_URL'] = 'http://127.0.0.1:8093'
    os.environ['EMBEDDINGS_NAME'] = 'Qwen/Qwen3-Embedding-0.6B'
    os.environ['EMBEDDINGS_MAX_QUERY_TOKENS'] = '0'
    os.environ['EMBEDDINGS_MAX_INPUT_TOKENS'] = '0'
    os.environ['EMBEDDINGS_BATCH_SIZE'] = '32'
    os.environ['EMBEDDINGS_CACHE_DIR'] = str(ROOT / 'devsupport/runtime/native-tokenizer-cache')
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['VECTOR_STORE'] = 'faiss'
    os.environ['DEVSUPPORT_GATEWAY_API_KEY'] = load_credential(ROOT / '.secrets/deepseek.dpapi')
    os.environ['INTERNAL_KEY'] = load_credential(ROOT / '.secrets/internal.dpapi')


def main() -> None:
    """Run one loopback process; credentials never enter a command argument or file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=7091)
    parser.add_argument('--role', choices=['api', 'worker', 'beat'], default='api')
    args = parser.parse_args()
    configure()
    if args.role != 'api':
        from docsgpt.app import celery
        command = (['worker', '--pool=solo', '--loglevel=INFO'] if args.role == 'worker'
                   else ['beat', '--loglevel=INFO'])
        celery.start(command)
        return
    import uvicorn
    uvicorn.run('docsgpt.asgi:asgi_app', host='127.0.0.1', port=args.port, access_log=False)


if __name__ == '__main__':
    main()
