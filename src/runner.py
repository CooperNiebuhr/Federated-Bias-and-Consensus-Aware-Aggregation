# src/runner.py

import itertools
import subprocess
import sys

def main():
    # Base configuration (what you described)
    dataset = "mnist"
    model = "cnn"
    num_users = 20          # K = 20 clients
    frac = 0.5              # 10 clients per round (0.5 * 20)
    epochs = 30             # global rounds
    local_ep = 10           # local epochs
    gpu = "0"               # use GPU 0
    iid = 0                 # non-IID; change to 1 if you want IID
    optimizer = "sgd"
    lr = 0.01               # adjust if needed
    local_bs = 10           # default from options.py
    partition_seed = 3       # Fixed seed for non-IID data partitioning

    # Hyperparameter grids
    lambdas = [1e-8, 1e-7, 1e-6]      # --lambda_reg
    betas   = [0.9, 0.95]             # --beta
    gammas  = [1.0, 2.0, 3.0]         # --gamma

    # “Seed pattern” – run each config with fixed seeds to reduce randomness
    seed = 20

    # Path to your main training script
    # (assuming you're running this from the repo root)
    federated_main = "src/federated_main.py"

    for lambda_reg, beta, gamma in itertools.product(lambdas, betas, gammas):
    
        cmd = [
            sys.executable, federated_main,
            "--dataset", dataset,
            "--model", model,
            "--num_users", str(num_users),
            "--frac", str(frac),
            "--epochs", str(epochs),
            "--local_ep", str(local_ep),
            "--local_bs", str(local_bs),
            "--iid", str(iid),
            "--optimizer", optimizer,
            "--lr", str(lr),
            "--seed", str(seed),
            "--partition_seed", str(partition_seed),
            

            # FedBaC-specific hyperparams:
            "--lambda_reg", str(lambda_reg),
            "--beta", str(beta),
            "--gamma", str(gamma),
        ]

        print("\n===================================================")
        print(f"Running: lambda={lambda_reg}, beta={beta}, gamma={gamma}, seed={seed}")
        print("Command:", " ".join(cmd))
        print("===================================================\n")

        # This will block until each run finishes
        subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
