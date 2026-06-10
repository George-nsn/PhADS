import os
import argparse
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser(description="Download ProstT5 model from HuggingFace")
    parser.add_argument("-o", "--output", default="./prostt5_model",
                        help="Output directory for the downloaded model (default: ./prostt5_model)")
    args = parser.parse_args()

    out_dir = os.path.abspath(args.output)
    print(f"Downloading ProstT5 model to: {out_dir}")
    snapshot_download(repo_id="Rostlab/ProstT5", local_dir=out_dir)
    print(f"✓ Download complete: {out_dir}")


if __name__ == "__main__":
    main()