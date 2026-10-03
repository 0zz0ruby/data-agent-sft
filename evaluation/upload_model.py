"""Upload the final checkpoint to a private Hugging Face model repository.

Authenticate with `hf auth login` first; never put tokens in source or arguments.
"""
import argparse
from pathlib import Path
from huggingface_hub import HfApi


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--repo-id', required=True, help='your-account/dataagent-qwen35-0.8b-sft')
    args = parser.parse_args()
    for name in ('model.safetensors', 'config.json', 'tokenizer.json', 'README.md'):
        if not (args.checkpoint / name).is_file():
            raise SystemExit(f'Missing checkpoint file: {name}')
    api = HfApi()
    identity = api.whoami()
    print(f"Authenticated as {identity['name']}")
    # First release stays private until upstream redistribution terms are reviewed.
    api.create_repo(repo_id=args.repo_id, repo_type='model', private=True, exist_ok=False)
    api.upload_folder(
        repo_id=args.repo_id,
        repo_type='model',
        folder_path=args.checkpoint,
        allow_patterns=['*.safetensors', '*.json', '*.jinja', 'README.md'],
        commit_message='Upload verified DataAgent SFT checkpoint and model card',
    )
    print(f'Uploaded private model repository: {args.repo_id}')


if __name__ == '__main__':
    main()
