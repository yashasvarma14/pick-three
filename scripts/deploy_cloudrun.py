"""
Deploy Pick Three to Google Cloud Run.

One-time setup:
    1. Install the Google Cloud CLI:   winget install Google.CloudSDK   (then open a new terminal)
    2. Log in:                         gcloud auth login
    3. Pick the project you use for Google Maps:
                                       gcloud config set project YOUR-PROJECT-ID

Deploy (re-run any time to push changes):
    python scripts/deploy_cloudrun.py

What it does
  * enables Cloud Run, Cloud Build and Artifact Registry in your project
  * gives the project's build account the "Cloud Run Builder" role (new projects need this)
  * builds the Dockerfile on Google's servers and deploys it as the "pick-three" service
  * copies your API keys from .streamlit/secrets.toml into the service's settings
    (through a temporary file that is deleted right after; the secrets file itself is
    never uploaded, see .gcloudignore)

Private by default: only your Google account can open it, through
    gcloud run services proxy pick-three --region us-central1 --port 8080
and then http://localhost:8080. Add --public to give it a shareable link.

Cost guards: scales to zero when idle (nothing runs, nothing billed), and at most one
instance runs at a time. Nothing redeploys on its own; it only changes when you run this.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SECRETS_FILE = ROOT / ".streamlit" / "secrets.toml"
SERVICE = "pick-three"
SETTINGS = ["ANTHROPIC_API_KEY", "GOOGLE_MAPS_API_KEY", "TICKETMASTER_API_KEY",
            "LLM_MODEL", "LLM_EFFORT", "RUNS_PER_DAY", "RUNS_PER_VISITOR_PER_DAY"]


def gcloud():
    path = shutil.which("gcloud")
    if not path:
        sys.exit("The Google Cloud CLI isn't installed. Run:  winget install Google.CloudSDK  (then open a new terminal)")
    return path


def run(args, capture=False):
    result = subprocess.run([gcloud(), *args], cwd=ROOT, text=True, capture_output=capture)
    if result.returncode != 0:
        if capture:
            print(result.stderr.strip())
        sys.exit(f"gcloud {' '.join(args[:3])} ... failed (see above).")
    return result.stdout.strip() if capture else None


def local_settings():
    try:
        data = tomllib.loads(SECRETS_FILE.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        sys.exit(f"Couldn't read {SECRETS_FILE}. Copy .streamlit/secrets.toml.example there and add your keys.")
    # Template placeholders ("sk-ant-...") don't count.
    return {k: str(data[k]) for k in SETTINGS
            if k in data and data[k] not in (None, "") and not str(data[k]).endswith("...")}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--region", default="us-central1")
    parser.add_argument("--public", action="store_true", help="anyone with the link can open it (default: private)")
    args = parser.parse_args()

    account = run(["auth", "list", "--filter=status:ACTIVE", "--format=value(account)"], capture=True)
    if not account:
        sys.exit("Not logged in. Run:  gcloud auth login")
    project = run(["config", "get-value", "project"], capture=True)
    if not project or project == "(unset)":
        sys.exit("No project selected. Run:  gcloud config set project YOUR-PROJECT-ID  (the one with your Maps key)")
    print(f"Account: {account}\nProject: {project}")

    settings = local_settings()
    if not args.public:
        # Only you can reach a private deploy, so give yourself room to try it out.
        settings.setdefault("RUNS_PER_VISITOR_PER_DAY", "50")
        settings.setdefault("RUNS_PER_DAY", "50")
    missing = [k for k in ["ANTHROPIC_API_KEY", "GOOGLE_MAPS_API_KEY"] if k not in settings]
    if missing:
        sys.exit(f"Missing in .streamlit/secrets.toml: {', '.join(missing)}")

    print("Enabling Cloud Run, Cloud Build and Artifact Registry (first time only takes a minute)...")
    run(["services", "enable", "run.googleapis.com", "cloudbuild.googleapis.com", "artifactregistry.googleapis.com"])

    # New projects build with the default compute account, which no longer gets the permissions
    # it needs (read the uploaded code, push the image, write logs). "Cloud Run Builder" is the
    # narrow role Google recommends for exactly that. Safe to repeat.
    number = run(["projects", "describe", project, "--format=value(projectNumber)"], capture=True)
    print("Making sure the build account can build (Cloud Run Builder role)...")
    run(["projects", "add-iam-policy-binding", project,
         f"--member=serviceAccount:{number}-compute@developer.gserviceaccount.com",
         "--role=roles/run.builder", "--condition=None", "--quiet"], capture=True)

    # Keys go in a temporary file so they never show up on a command line or in the terminal.
    fd, env_file = tempfile.mkstemp(suffix=".yaml")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for key, value in settings.items():
                f.write(f"{key}: {json.dumps(value)}\n")  # a JSON string is valid, safely quoted YAML
        print("Building and deploying (first build ~5-10 minutes)...")
        run(["run", "deploy", SERVICE, "--source", ".", "--region", args.region,
             "--memory", "2Gi", "--cpu", "1", "--timeout", "300",
             "--min-instances", "0", "--max-instances", "1", "--concurrency", "10",
             # Extra CPU only while the container starts (loading PyTorch + the embedding
             # model), so the first visit after an idle spell wakes up faster.
             "--cpu-boost",
             "--env-vars-file", env_file,
             "--allow-unauthenticated" if args.public else "--no-allow-unauthenticated",
             "--quiet"])
    finally:
        os.remove(env_file)

    url = run(["run", "services", "describe", SERVICE, "--region", args.region, "--format=value(status.url)"],
              capture=True)
    print("\nDone.")
    if args.public:
        print(f"  Live (public): {url}")
    else:
        print("  Private: only your Google account can open it. To view it:")
        print(f"    gcloud run services proxy {SERVICE} --region {args.region} --port 8080")
        print("    then open http://localhost:8080")
        print("  To make it public later:  python scripts/deploy_cloudrun.py --public")


if __name__ == "__main__":
    main()
