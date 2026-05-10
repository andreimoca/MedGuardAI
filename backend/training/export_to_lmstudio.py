"""Move a fine-tuned GGUF into LM Studio's local models directory and (optionally)
push the artifacts to a Hugging Face Hub repo so teammates can pull them.

Usage:
    # Copy a GGUF you downloaded from Colab into LM Studio:
    python export_to_lmstudio.py copy --gguf path/to/medguard-gemma-3-4b-q4_k_m.gguf

    # Push everything to a private HF Hub repo:
    python export_to_lmstudio.py push \
        --repo your-username/medguardai-gemma-3-4b \
        --adapter-dir adapters/medguard-sft \
        --gguf-dir adapters/medguard-gguf \
        --token hf_xxx [--public]

The Colab/Kaggle notebook (`finetune_gemma_lora.ipynb`) is the recommended path
for actually producing the GGUF — your 6 GB local GPU isn't enough headroom to
merge + quantize Gemma-3-4b once Brave/Teams/etc. are running. This script
assumes the heavy work happened in the cloud and you're now bringing the
artifact home.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path


def lmstudio_models_dir() -> Path:
    """LM Studio's default models directory across platforms.

    Override with MEDGUARD_LMSTUDIO_MODELS_DIR if your install is non-standard.
    """
    override = os.environ.get("MEDGUARD_LMSTUDIO_MODELS_DIR")
    if override:
        return Path(override)
    if sys.platform == "win32":
        return Path(os.path.expandvars(r"%USERPROFILE%\.lmstudio\models"))
    if sys.platform == "darwin":
        return Path.home() / ".lmstudio" / "models"
    return Path.home() / ".lmstudio" / "models"


def cmd_copy(args: argparse.Namespace) -> int:
    src = Path(args.gguf).resolve()
    if not src.exists():
        print(f"error: {src} does not exist")
        return 1
    if not src.suffix.lower() == ".gguf":
        print(f"warning: {src.name} is not a .gguf file (continuing anyway)")

    target_root = lmstudio_models_dir()
    if not target_root.exists():
        print(f"error: LM Studio models directory not found at {target_root}")
        print("Set MEDGUARD_LMSTUDIO_MODELS_DIR to the actual path, or open LM Studio "
              "once and check Settings → 'My Models' to confirm where it stores models.")
        return 1

    # LM Studio expects models under a publisher/model/quant nested folder.
    publisher = args.publisher or "medguard"
    model_name = args.model_name or "medguard-gemma-3-4b"
    target_dir = target_root / publisher / model_name
    target_dir.mkdir(parents=True, exist_ok=True)
    dest = target_dir / src.name
    shutil.copy2(src, dest)

    print(f"copied:\n  {src}\n  -> {dest}")
    print(f"\nNow open LM Studio, load the model, and update .env:")
    print(f"  LOCAL_LLM_MODEL={publisher}/{model_name}")
    print("Restart the backend and the agent will use the fine-tuned weights.")
    return 0


def cmd_push(args: argparse.Namespace) -> int:
    try:
        from huggingface_hub import HfApi, create_repo, login
    except ImportError:
        print("error: install huggingface_hub first  (pip install huggingface_hub)")
        return 1

    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        print("error: provide --token or set HF_TOKEN env var")
        return 1
    login(token=token)

    create_repo(args.repo, private=not args.public, exist_ok=True)
    api = HfApi()

    if args.adapter_dir:
        api.upload_folder(
            repo_id=args.repo,
            folder_path=args.adapter_dir,
            path_in_repo="sft-adapter",
        )
        print(f"pushed LoRA adapter from {args.adapter_dir}")

    if args.gguf_dir:
        api.upload_folder(
            repo_id=args.repo,
            folder_path=args.gguf_dir,
            path_in_repo="gguf",
        )
        print(f"pushed GGUF from {args.gguf_dir}")

    visibility = "public" if args.public else "private"
    print(f"\ndone — {visibility} repo at https://huggingface.co/{args.repo}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_copy = sub.add_parser("copy", help="copy a GGUF into LM Studio's models folder")
    p_copy.add_argument("--gguf", required=True, help="path to the .gguf file")
    p_copy.add_argument("--publisher", default="medguard")
    p_copy.add_argument("--model-name", default="medguard-gemma-3-4b")
    p_copy.set_defaults(func=cmd_copy)

    p_push = sub.add_parser("push", help="push artifacts to Hugging Face Hub")
    p_push.add_argument("--repo", required=True, help="username/repo")
    p_push.add_argument("--adapter-dir", help="local LoRA adapter dir")
    p_push.add_argument("--gguf-dir", help="local dir containing the .gguf file(s)")
    p_push.add_argument("--token", help="HF write token (or set HF_TOKEN env)")
    p_push.add_argument("--public", action="store_true", help="make the repo public")
    p_push.set_defaults(func=cmd_push)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
