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
    if name in ["ddos-http", "ddos_http", "ddos"]:
        return "ddos_http"
    return name

def run_command(cmd, cwd=SRC_DIR):
    print(f"\n[+] Executing: {' '.join(cmd)}\n")
    try:
        proc = subprocess.run(cmd, cwd=cwd)
        return proc.returncode
    except KeyboardInterrupt:
        print("\n[!] Execution stopped by user.")
        return 1

def train_model(dataset="edge_iot", epochs=10, exp_name=None, network="STANNet"):
    dataset = clean_dataset_name(dataset)
    if not exp_name:
        exp_name = f"intra_{dataset.replace('_', '-')}_{network.lower()}"
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
        "--network", network,
        "--approach", "scratch",
        "--seed", "1"
    ]
    code = run_command(cmd)
    if code == 0:
        print("\n[✓] Training complete! Now computing metrics...")
        compute_metrics(exp_name)

def transfer_learning(source="edge_iot", target="ton_iot", epochs=10, network="STANNet", approach="dann"):
    source = clean_dataset_name(source)
    target = clean_dataset_name(target)
    exp_name = f"transfer_{source}_to_{target}_{network.lower()}_{approach}"
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
        "--network", network,
        "--approach", approach,
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
    # Search for latest matching result dir
    results_base = os.path.join(BASE_DIR, "results")
    matched_dir = None
    if os.path.exists(results_base):
        matches = [os.path.join(results_base, d) for d in os.listdir(results_base) if exp_name in d]
        if matches:
            matches.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            matched_dir = matches[0]
    view_results(matched_dir)

def view_results(exp_dir=None):
    cmd = [VENV_PYTHON, "view_results.py"]
    if exp_dir:
        cmd.append(exp_dir)
    run_command(cmd)

def predict_attacks(input_target="edge_iot", samples=500):
    cmd = [VENV_PYTHON, "predict.py", input_target, "--samples", str(samples)]
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
        print(" [5] Auto-Detect / Predict Attacks (Dataset or PCAP)")
        print(" [6] Generate Sample IoT Dataset")
        print(" [7] Exit")
        print("=" * 55)
        
        choice = input("Select an option (1-7): ").strip()
        
        if choice == "1":
            print("\nAvailable datasets: edge_iot, ton_iot, iot_nidd, sample_iot, ddos_http")
            dset = input("Enter dataset name (default: edge_iot): ").strip() or "edge_iot"
            ep = input("Enter number of epochs (default: 5): ").strip() or "5"
            net = input("Choose Network [1] STANNet (Recommended SOTA) [2] Lopez17CNN (default: 1): ").strip()
            net_choice = "Lopez17CNN" if net == "2" else "STANNet"
            train_model(dataset=dset, epochs=int(ep), network=net_choice)
            
        elif choice == "2":
            print("\nAvailable datasets: edge_iot, ton_iot, iot_nidd, sample_iot, ddos_http")
            src = input("Enter Source dataset (default: edge_iot): ").strip() or "edge_iot"
            tgt = input("Enter Target dataset (default: ton_iot): ").strip() or "ton_iot"
            ep = input("Enter epochs per task (default: 5): ").strip() or "5"
            net = input("Choose Network [1] STANNet (Recommended SOTA) [2] Lopez17CNN (default: 1): ").strip()
            net_choice = "Lopez17CNN" if net == "2" else "STANNet"
            app = input("Choose Approach [1] DANN (Domain-Adversarial, SOTA) [2] Fine-Tuning [3] BiC (default: 1): ").strip()
            if app == "2":
                app_choice = "jointft"
            elif app == "3":
                app_choice = "bic"
            else:
                app_choice = "dann"
            transfer_learning(source=src, target=tgt, epochs=int(ep), network=net_choice, approach=app_choice)
            
        elif choice == "3":
            exp = input("Enter experiment name (default: intra_edge-iot): ").strip() or "intra_edge-iot"
            compute_metrics(exp)
            
        elif choice == "4":
            view_results()
            
        elif choice == "5":
            target = input("Enter dataset name (e.g. edge_iot, ddos_http) OR path to .pcap/.parquet file: ").strip() or "edge_iot"
            num_s = input("Enter max flows to analyze (default: 500): ").strip() or "500"
            predict_attacks(input_target=target, samples=int(num_s))
            
        elif choice == "6":
            run_command([VENV_PYTHON, "generate_sample_data.py"])
            
        elif choice == "7":
            print("Exiting...")
            break
        else:
            print("[!] Invalid option. Please choose between 1 and 7.")

def main():
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg == "train":
            dset = sys.argv[2] if len(sys.argv) > 2 else "edge_iot"
            ep = int(sys.argv[3]) if len(sys.argv) > 3 else 5
            net = sys.argv[4] if len(sys.argv) > 4 else "STANNet"
            train_model(dataset=dset, epochs=ep, network=net)
        elif arg in ["transfer", "adapt"]:
            src = sys.argv[2] if len(sys.argv) > 2 else "edge_iot"
            tgt = sys.argv[3] if len(sys.argv) > 3 else "ton_iot"
            ep = int(sys.argv[4]) if len(sys.argv) > 4 else 5
            net = sys.argv[5] if len(sys.argv) > 5 else "STANNet"
            app = sys.argv[6] if len(sys.argv) > 6 else "dann"
            transfer_learning(source=src, target=tgt, epochs=ep, network=net, approach=app)
        elif arg in ["predict", "detect"]:
            target = sys.argv[2] if len(sys.argv) > 2 else "edge_iot"
            num_s = int(sys.argv[3]) if len(sys.argv) > 3 else 500
            predict_attacks(input_target=target, samples=num_s)
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
            print("  python run.py train [dataset] [epochs] [network]")
            print("  python run.py transfer [src] [tgt] [epochs] [network] [approach]")
            print("  python run.py predict [dataset_or_pcap] [samples]")
            print("  python run.py eval [exp_name]")
            print("  python run.py results [exp_dir]")
    else:
        interactive_menu()

if __name__ == "__main__":
    main()
