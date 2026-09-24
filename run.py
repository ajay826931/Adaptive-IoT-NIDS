import os
import sys
import subprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
VENV_PYTHON = os.path.join(BASE_DIR, "venv", "Scripts", "python.exe")

if not os.path.exists(VENV_PYTHON):
    VENV_PYTHON = sys.executable

def clean_dataset_name(name):
    name = name.strip().replace("'", "").replace('"', '').replace('!', '')
    if name.endswith(".parquet"):
        name = name[:-8]
    if name.endswith("_dwn10p"):
        name = name[:-7]
    if name in ["edge-iiot", "edge_iiot", "edge-iot"]:
        return "edge_iot"
    if name in ["ton-iot", "ton_iot"]:
        return "ton_iot"
    if name in ["iot-nidd", "iot_nidd"]:
        return "iot_nidd"
    if name in ["sample-iot", "sample_iot"]:
        return "sample_iot"
    return name

def run_command(cmd, cwd=SRC_DIR):
    print(f"\n[+] Executing: {' '.join(cmd)}\n")
    try:
        proc = subprocess.run(cmd, cwd=cwd)
        return proc.returncode
    except KeyboardInterrupt:
        print("\n[!] Execution stopped by user.")
        return 1

def train_model(dataset="edge_iot", epochs=10, exp_name=None):
    dataset = clean_dataset_name(dataset)
    if not exp_name:
        exp_name = f"intra_{dataset.replace('_', '-')}"
    cmd = [
        VENV_PYTHON, "main.py",
        "--exp-name", exp_name,
        "--results-path", "../results/",
        "--datasets", dataset,
        "--num-tasks", "1",
        "--fields", "PL", "IAT", "DIR", "WIN",
        "--num-pkts", "10",
        "--batch-size", "64",
        "--nepochs", str(epochs),
        "--save-models",
        "--network", "Lopez17CNN",
        "--approach", "scratch",
        "--seed", "1"
    ]
    code = run_command(cmd)
    if code == 0:
        print("\n[✓] Training complete! Now computing metrics...")
        compute_metrics(exp_name)

def transfer_learning(source="edge_iot", target="ton_iot", epochs=10):
    source = clean_dataset_name(source)
    target = clean_dataset_name(target)
    exp_name = f"transfer_{source}_to_{target}"
    cmd = [
        VENV_PYTHON, "main.py",
        "--exp-name", exp_name,
        "--results-path", "../results/",
        "--datasets", source, target,
        "--num-tasks", "2",
        "--fields", "PL", "IAT", "DIR", "WIN",
        "--num-pkts", "10",
        "--batch-size", "64",
        "--nepochs", str(epochs),
        "--save-models",
        "--network", "Lopez17CNN",
        "--approach", "finetuning",
        "--seed", "1"
    ]
    code = run_command(cmd)
    if code == 0:
        print("\n[✓] Transfer learning complete! Now computing metrics...")
        compute_metrics(exp_name)

def compute_metrics(exp_name="intra_edge-iot"):
    cmd = [
        VENV_PYTHON, "compute_metrics.py",
        "--exp-name", exp_name,
        "--partial-exp-name",
        "--results-path", "../results/",
        "--yes"
    ]
    run_command(cmd)
    # Search for matching result dir
    results_base = os.path.join(BASE_DIR, "results")
    matched_dir = None
    if os.path.exists(results_base):
        for d in os.listdir(results_base):
            if exp_name in d:
                matched_dir = os.path.join(results_base, d)
                break
    view_results(matched_dir)

def view_results(exp_dir=None):
    cmd = [VENV_PYTHON, "view_results.py"]
    if exp_dir:
        cmd.append(exp_dir)
    run_command(cmd)

def interactive_menu():
    while True:
        print("\n" + "=" * 55)
        print("          ADAPTIVE IoT-NIDS : EASY LAUNCHER          ")
        print("=" * 55)
        print(" [1] Train Model (Single Dataset)")
        print(" [2] Transfer / Incremental Learning (2 Datasets)")
        print(" [3] Compute Metrics for an Experiment")
        print(" [4] View Latest Experiment Results (Table)")
        print(" [5] Generate Sample IoT Dataset")
        print(" [6] Exit")
        print("=" * 55)
        
        choice = input("Select an option (1-6): ").strip()
        
        if choice == "1":
            print("\nAvailable datasets: edge_iot, ton_iot, iot_nidd, sample_iot")
            dset = input("Enter dataset name (default: edge_iot): ").strip() or "edge_iot"
            ep = input("Enter number of epochs (default: 5): ").strip() or "5"
            train_model(dataset=dset, epochs=int(ep))
            
        elif choice == "2":
            src = input("Enter Source dataset (default: edge_iot): ").strip() or "edge_iot"
            tgt = input("Enter Target dataset (default: ton_iot): ").strip() or "ton_iot"
            ep = input("Enter epochs per task (default: 5): ").strip() or "5"
            transfer_learning(source=src, target=tgt, epochs=int(ep))
            
        elif choice == "3":
            exp = input("Enter experiment name (default: intra_edge-iot): ").strip() or "intra_edge-iot"
            compute_metrics(exp)
            
        elif choice == "4":
            view_results()
            
        elif choice == "5":
            run_command([VENV_PYTHON, "generate_sample_data.py"])
            
        elif choice == "6":
            print("Exiting...")
            break
        else:
            print("[!] Invalid option. Please choose between 1 and 6.")

def main():
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg == "train":
            dset = sys.argv[2] if len(sys.argv) > 2 else "edge_iot"
            ep = int(sys.argv[3]) if len(sys.argv) > 3 else 5
            train_model(dataset=dset, epochs=ep)
        elif arg in ["eval", "metrics"]:
            exp = sys.argv[2] if len(sys.argv) > 2 else "intra_edge-iot"
            compute_metrics(exp)
        elif arg == "results":
            target = sys.argv[2] if len(sys.argv) > 2 else None
            view_results(target)
        elif arg == "sample":
            run_command([VENV_PYTHON, "generate_sample_data.py"])
        else:
            print(f"Unknown command '{arg}'.")
            print("Usage:")
            print("  python run.py")
            print("  python run.py train [dataset] [epochs]")
            print("  python run.py eval [exp_name]")
            print("  python run.py results [exp_dir]")
    else:
        interactive_menu()

if __name__ == "__main__":
    main()
