"""
Deploy Pick Three to a Hugging Face Space (Docker).

One-time setup (in a terminal, with the project's .venv):
    hf auth login          # paste a token with "write" access from huggingface.co/settings/tokens

Deploy (re-run any time to push changes):
    python scripts/deploy_hf.py --space YOUR-HF-USERNAME/pick-three

The Space is PRIVATE by default: only you, signed in to Hugging Face, can open it.
Add --public when you want to share the link (e.g. on a resume).

Nothing updates on its own: the Space only changes when you run this script again.

Add --with-secrets to also copy the API keys from .streamlit/secrets.toml into the
Space's private secrets (they're stored encrypted by Hugging Face and never uploaded
as a file). Without it, add them by hand: Space -> Settings -> Variables and secrets.
"""

import argparse
import sys
import tomllib
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError

ROOT = Path(__file__).resolve().parent.parent
SPACE_CARD = ROOT / "deploy" / "huggingface" / "README.md"
SECRETS_FILE = ROOT / ".streamlit" / "secrets.toml"

# Same idea as .dockerignore: code and data only. The secrets file is never uploaded.
IGNORE = [
    ".streamlit/secrets.toml", ".env", ".venv/*", ".git/*", ".claude/*", "notebooks/*", "tests/*", "docs/*",
    "*/__pycache__/*", "*.pyc", ".pytest_cache/*", "README.md",
]
SECRET_KEYS = ["ANTHROPIC_API_KEY", "GOOGLE_MAPS_API_KEY", "TICKETMASTER_API_KEY"]
SETTINGS = ["LLM_MODEL", "LLM_EFFORT", "RUNS_PER_DAY", "RUNS_PER_VISITOR_PER_DAY"]


def local_secrets():
    try:
        data = tomllib.loads(SECRETS_FILE.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return {}
    # Placeholders from the template ("sk-ant-...") don't count.
    return {k: str(v) for k, v in data.items() if v not in (None, "") and not str(v).endswith("...")}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--space", required=True, help="USERNAME/space-name, e.g. yashas/pick-three")
    parser.add_argument("--with-secrets", action="store_true", help="copy API keys from .streamlit/secrets.toml")
    parser.add_argument("--public", action="store_true", help="make the Space public (default: private)")
    args = parser.parse_args()

    api = HfApi()
    try:
        user = api.whoami()["name"]
    except Exception:
        sys.exit("Not logged in to Hugging Face. Run:  hf auth login   (needs a token with write access)")
    print(f"Logged in as {user}")

    api.create_repo(args.space, repo_type="space", space_sdk="docker", private=not args.public, exist_ok=True)
    # Applies to an existing Space too, so the visibility always matches what you asked for.
    api.update_repo_settings(args.space, repo_type="space", private=not args.public)
    print(f"Space ready ({'public' if args.public else 'private: only you can see it'}): "
          f"https://huggingface.co/spaces/{args.space}")

    if args.with_secrets:
        found = local_secrets()
        for key in SECRET_KEYS:
            if key in found:
                api.add_space_secret(args.space, key, found[key])
                print(f"  secret set: {key}")
        for key in SETTINGS:
            if key in found:
                api.add_space_variable(args.space, key, found[key])
                print(f"  setting set: {key} = {found[key]}")
        missing = [k for k in SECRET_KEYS[:2] if k not in found]
        if missing:
            print(f"  Not found locally (add them in the Space settings): {', '.join(missing)}")

    print("Uploading code (the secrets file is excluded)...")
    try:
        api.upload_folder(folder_path=str(ROOT), repo_id=args.space, repo_type="space",
                          ignore_patterns=IGNORE, commit_message="Deploy Pick Three")
        api.upload_file(path_or_fileobj=str(SPACE_CARD), path_in_repo="README.md", repo_id=args.space,
                        repo_type="space", commit_message="Space card")
    except HfHubHTTPError as exc:
        sys.exit(f"Upload failed: {exc}")

    owner, name = args.space.split("/", 1)
    print("\nDone. Hugging Face is building the image now (first build takes ~5-10 minutes).")
    print(f"  Build logs: https://huggingface.co/spaces/{args.space}?logs=build")
    print(f"  Live site:  https://{owner.lower()}-{name.lower().replace('_', '-')}.hf.space")
    if not args.public:
        print("  (Private: open it while signed in to huggingface.co. Re-run with --public to share it.)")


if __name__ == "__main__":
    main()
