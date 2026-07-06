import os
import sys
import argparse
import subprocess

def run_foldseek(input_path, out_dir, db_path):
    os.makedirs(out_dir, exist_ok=True)
    
    result_tsv = os.path.join(out_dir, "result.tsv")
    tmp_dir = os.path.join(out_dir, "tmp")
    
    if not os.path.exists(input_path):
        print(f"ERROR: input path does not exist: {input_path}", file=sys.stderr)
        sys.exit(1)
        
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
    print("Starting Foldseek search...")
    print(f"Input path: {input_path}")
    print(f"Database: {db_path}")
    print(f"Output file: {result_tsv}")
    print(f"Temporary directory: {tmp_dir}")
    print("=" * 50)
    
    try:
        subprocess.run(cmd, check=True)
        print("-" * 50)
        print("Foldseek completed successfully; sorting hits by descending probability...")
        
        if os.path.exists(result_tsv):
            with open(result_tsv, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            if len(lines) > 1:
                header = lines[0]
                data_lines = lines[1:]
                
                def get_prob_val(line):
                    columns = line.strip().split('\t')
                    if len(columns) > 2:
                        try:
                            return float(columns[2])
                        except ValueError:
                            return 0.0
                    return 0.0
                
                data_lines.sort(key=get_prob_val, reverse=True)
                
                with open(result_tsv, 'w', encoding='utf-8') as f:
                    f.writelines([header] + data_lines)
                
            print("Sorting completed")
            print(f"Result saved to: {result_tsv}")
            
    except subprocess.CalledProcessError as e:
        print(f"\nERROR: Foldseek returned non-zero exit code {e.returncode}", file=sys.stderr)
        sys.exit(e.returncode)
    except FileNotFoundError:
        print("\nERROR: the 'foldseek' executable was not found in PATH", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Foldseek search and sort hits by descending probability")
    parser.add_argument("-i", "--input", required=True, help="Input CIF/PDB file path, or a directory containing structure files")
    parser.add_argument("-o", "--outdir", required=True, help="Output directory")
    parser.add_argument("-d", "--db", required=True, help="Foldseek database path/prefix")
    
    args = parser.parse_args()
    run_foldseek(args.input, args.outdir, args.db)