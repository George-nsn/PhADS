import os
import argparse

def download_models(output_dir, repo_ids):
    """Download the required Hugging Face repositories into the target directory."""
    from huggingface_hub import snapshot_download
    from tqdm import tqdm

    os.makedirs(output_dir, exist_ok=True)

    for repo_id in tqdm(repo_ids, desc="Downloading model repositories", unit="repo"):
        folder_name = repo_id.split('/')[-1]
        local_dir = os.path.join(output_dir, folder_name)

        print(f"Starting repository download: {repo_id}")
        print(f"Destination: {local_dir}")

        try:
            snapshot_download(
                repo_id=repo_id,
                local_dir=local_dir,
                local_dir_use_symlinks=False,
                resume_download=True,
            )
            print(f"Repository download completed: {repo_id}\n")
        except Exception as e:
            print(f"Repository download failed: {repo_id}; error={e}\n")

def main():
    parser = argparse.ArgumentParser(description="Download the required Hugging Face model repositories")
    parser.add_argument(
        "-o", "--output",
        type=str,
        required=True,
        help="Root directory used to store downloaded model repositories"
    )
    args = parser.parse_args()

    target_repos = [
        "facebook/esm2_t33_650M_UR50D",
        "Rostlab/prot-t5-xl-uniref50-enc-onnx"
    ]

    download_models(args.output, target_repos)

if __name__ == "__main__":
    main()