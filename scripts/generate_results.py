import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate
import yaml
from pathlib import Path
import sys

# Add the project root to the Python path
project_root = str(Path(__file__).resolve().parents[1])
sys.path.append(project_root)

def load_yaml(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def generate_results():
    # Find the latest multirun directory
    log_dir = Path(project_root) / "logs" / "train" / "multiruns"
    latest_run = max(log_dir.glob('*'), key=lambda p: p.stat().st_mtime)

    data = []
    for exp_dir in latest_run.glob('[0-9]*'):
        hparams_file = exp_dir / "csv" / "version_0" / "hparams.yaml"
        if hparams_file.exists():
            hparams = load_yaml(hparams_file)
            metrics_file = exp_dir / "csv" / "version_0" / "metrics.csv"
            if metrics_file.exists():
                metrics = pd.read_csv(metrics_file)
                val_acc = metrics['val/acc'].max()
                val_loss = metrics['val/loss'].min()
                
                # Add validation metrics to hparams
                hparams['val_acc'] = val_acc
                hparams['val_loss'] = val_loss
                
                data.append(hparams)

    df = pd.DataFrame(data)
    print(df)
    
    # Generate table
    table = tabulate(df, headers='keys', tablefmt='pipe', floatfmt='.4f')
    with open('results_table.md', 'w') as f:
        f.write(table)
    
    # Generate plot
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(df['val_loss'])
    plt.title('Validation Loss')
    plt.xlabel('Run')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(df['val_acc'])
    plt.title('Validation Accuracy')
    plt.xlabel('Run')
    plt.ylabel('Accuracy')
    
    plt.tight_layout()
    plt.savefig('combined_metrics_plot.png')

    # Load and print optimization results
    opt_results_file = latest_run / "optimization_results.yaml"
    if opt_results_file.exists():
        opt_results = load_yaml(opt_results_file)
        print("Best hyperparameters:")
        print(yaml.dump(opt_results['best_params'], default_flow_style=False))
        print(f"Best validation accuracy: {opt_results['best_value']:.4f}")

if __name__ == "__main__":
    generate_results()