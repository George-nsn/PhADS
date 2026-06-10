import os
import sys
import argparse
import subprocess

def run_foldseek(input_path, out_dir, db_path):
    # 自动创建输出目录
    os.makedirs(out_dir, exist_ok=True)
    
    result_tsv = os.path.join(out_dir, "result.tsv")
    tmp_dir = os.path.join(out_dir, "tmp")
    
    if not os.path.exists(input_path):
        print(f"错误: 输入路径 '{input_path}' 不存在，请检查路径是否正确。", file=sys.stderr)
        sys.exit(1)
        
    # 构建命令：将 prob 调至 target 后面（即第 3 列）
    cmd = [
        "foldseek", "easy-search",
        input_path,
        db_path,
        result_tsv,
        tmp_dir,
        "--format-mode", "4",
        "--format-output", "query,target,prob,fident,alnlen,mismatch,gapopen,qstart,qend,tstart,tend,evalue,bits,lddt,alntmscore"
    ]
    
    print("=" * 50)
    print("开始运行 Foldseek 比对...")
    print(f"输入路径:   {input_path}")
    print(f"数据库:     {db_path}")
    print(f"输出文件:   {result_tsv}")
    print(f"临时目录:   {tmp_dir}")
    print("=" * 50)
    
    try:
        # 运行 Foldseek
        subprocess.run(cmd, check=True)
        print("-" * 50)
        print("比对成功，正在按 prob 降序排序...")
        
        # 后处理：读取文件并排序
        if os.path.exists(result_tsv):
            with open(result_tsv, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # 只有在行数大于 1 时（有表头 + 至少一行数据）才需要排序
            if len(lines) > 1:
                header = lines[0]       # 第一行直接作为表头
                data_lines = lines[1:]  # 剩下的行作为数据行
                
                # 排序函数：将第 3 列（索引 2）转为 float 进行比较
                def get_prob_val(line):
                    columns = line.strip().split('\t')
                    if len(columns) > 2:
                        try:
                            return float(columns[2])
                        except ValueError:
                            return 0.0
                    return 0.0
                
                # 降序排序
                data_lines.sort(key=get_prob_val, reverse=True)
                
                # 写回原文件，确保表头 header 仍然在最顶端
                with open(result_tsv, 'w', encoding='utf-8') as f:
                    f.writelines([header] + data_lines)
                
            print("排序完成！")
            print(f"结果已保存至: {result_tsv}")
            
    except subprocess.CalledProcessError as e:
        print(f"\n运行失败: Foldseek 返回了错误代码 {e.returncode}。", file=sys.stderr)
        sys.exit(e.returncode)
    except FileNotFoundError:
        print("\n错误: 未在系统中找到 'foldseek' 命令。请确保其已被正确安装。", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Foldseek 自动化比对与 Prob 降序重排脚本")
    parser.add_argument("-i", "--input", required=True, help="输入 CIF/PDB 文件路径，或包含结构文件的目录")
    parser.add_argument("-o", "--outdir", required=True, help="输出目录")
    parser.add_argument("-d", "--db", required=True, help="Foldseek 数据库的路径及前缀")
    
    args = parser.parse_args()
    run_foldseek(args.input, args.outdir, args.db)