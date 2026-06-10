import os
import argparse
from huggingface_hub import snapshot_download

def download_models(output_dir, repo_ids):
    """
    下载指定的 Hugging Face 仓库到目标目录
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    for repo_id in repo_ids:
        # 使用仓库名（排除作者名）或完整路径作为子文件夹名称，避免多个模型文件混杂
        folder_name = repo_id.split('/')[-1]
        local_dir = os.path.join(output_dir, folder_name)

        print(f"正在下载仓库: {repo_id} ...")
        print(f"保存路径: {local_dir}")

        try:
            # snapshot_download 用于下载整个仓库
            # local_dir_use_symlinks=False 可以确保直接下载真实文件，而不是创建指向缓存的符号链接
            snapshot_download(
                repo_id=repo_id,
                local_dir=local_dir,
                local_dir_use_symlinks=False,
                resume_download=True  # 支持断点续传
            )
            print(f"下载完成: {repo_id}\n")
        except Exception as e:
            print(f"下载 {repo_id} 时发生错误: {e}\n")

def main():
    parser = argparse.ArgumentParser(description="下载指定的 Hugging Face 模型仓库")
    parser.add_argument(
        "-o", "--output",
        type=str,
        required=True,
        help="指定模型下载的保存根目录"
    )
    args = parser.parse_args()

    # 需要下载的目标仓库列表
    target_repos = [
        "facebook/esm2_t33_650M_UR50D",
        "Rostlab/prot-t5-xl-uniref50-enc-onnx"
    ]

    download_models(args.output, target_repos)

if __name__ == "__main__":
    main()