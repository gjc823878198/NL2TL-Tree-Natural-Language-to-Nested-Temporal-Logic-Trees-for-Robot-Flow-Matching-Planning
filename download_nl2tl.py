"""
Download the NL2TL repo (code) AND the NL2TL dataset sample file (data) to
the local `data/` folder. Fully offline after this step.

Sources (per NL2TL/README.md):
  - Code:    https://github.com/yongchao98/NL2TL
  - Dataset: Google Drive folder
             https://drive.google.com/drive/folders/10F-qyOhpqEi83o9ZojymqRPUtwzSOcfq
  - Sample NL-TL file (smaller, used for the demo):
             https://drive.google.com/file/d/1f-wQ8AKInlTpXTYKwICRC0eZ-JKjAefh
"""
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/yongchao98/NL2TL.git"
GDRIVE_SAMPLE_FILE_ID = "1f-wQ8AKInlTpXTYKwICRC0eZ-JKjAefh"   # sample NL-TL pairs
GDRIVE_FULL_FOLDER_ID = "10F-qyOhpqEi83o9ZojymqRPUtwzSOcfq"   # full dataset folder

HERE = Path(__file__).parent
DATA_DIR = HERE / "data"
DATA_DIR.mkdir(exist_ok=True)
REPO_DEST = DATA_DIR / "NL2TL_repo"
DATASET_DEST = DATA_DIR / "nl2tl_dataset"


def clone_repo():
    if REPO_DEST.exists():
        print(f"[1/2] repo already at {REPO_DEST}")
        return
    print(f"[1/2] git clone {REPO_URL} -> {REPO_DEST}")
    try:
        subprocess.run(
            ["git", "clone", "--depth=1", REPO_URL, str(REPO_DEST)],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"  ERROR: git clone failed: {e}", file=sys.stderr)
        sys.exit(1)


def download_dataset_sample():
    """Pull the small sample NL-TL pair file via gdown (no Google login needed)."""
    DATASET_DEST.mkdir(exist_ok=True)
    out_path = DATASET_DEST / "nl_tl_sample.json"
    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"[2/2] sample dataset already at {out_path} ({out_path.stat().st_size:,} bytes)")
        return out_path
    try:
        import gdown
    except ImportError:
        print("  gdown not installed; run: pip install gdown", file=sys.stderr)
        return None

    print(f"[2/2] downloading sample dataset from Google Drive ...")
    try:
        gdown.download(id=GDRIVE_SAMPLE_FILE_ID, output=str(out_path), quiet=False)
    except Exception as e:
        print(f"  WARNING: gdown failed: {e}", file=sys.stderr)
        print(f"  You can download manually:")
        print(f"    https://drive.google.com/file/d/{GDRIVE_SAMPLE_FILE_ID}/view")
        print(f"  and save it to: {out_path}")
        return None
    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"  saved to {out_path} ({out_path.stat().st_size:,} bytes)")
        return out_path
    return None


def list_local_data():
    print("\nLocal data files (any size):")
    found = False
    for root in [REPO_DEST, DATASET_DEST]:
        if not root.exists():
            continue
        for p in sorted(root.rglob("*")):
            if not p.is_file():
                continue
            if p.suffix.lower() in (".csv", ".json", ".jsonl", ".tsv", ".txt"):
                size = p.stat().st_size
                print(f"  {p.relative_to(HERE)}  ({size:,} bytes)")
                found = True
    if not found:
        print("  (none yet)")


def main():
    clone_repo()
    download_dataset_sample()
    list_local_data()

    print("\nNext step — convert to JSONL with parse trees:")
    print(f"  python3 convert_dataset.py data/nl2tl_dataset/nl_tl_sample.json "
          f"data/nl2tl_converted.jsonl --limit 100")
    print("\nFor the FULL dataset (~28k pairs), manually open this folder in a browser")
    print("and place files into data/nl2tl_dataset/:")
    print(f"  https://drive.google.com/drive/folders/{GDRIVE_FULL_FOLDER_ID}")


if __name__ == "__main__":
    main()
