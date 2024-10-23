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
    test_accuracies_step = {}
    test_accuracies_epoch = {}
    
    for exp_dir in latest_run.glob('[0-9]*'):
        exp_num = exp_dir.name
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
                
                # Extract test accuracy for plotting
                test_acc_step = metrics[['step', 'test/acc']].dropna()
                test_acc_epoch = metrics[['epoch', 'test/acc']].dropna()
                if not test_acc_step.empty:
                    test_accuracies_step[exp_num] = test_acc_step
                if not test_acc_epoch.empty:
                    test_accuracies_epoch[exp_num] = test_acc_epoch

    df = pd.DataFrame(data)
    
    # Generate hyperparameters table
    hparams_table = tabulate(df, headers='keys', tablefmt='pipe', floatfmt='.4f')
    with open('hyperparameters_table.md', 'w') as f:
        f.write(hparams_table)
    
    # Generate test accuracy plot (step-wise)
    plt.figure(figsize=(10, 6))
    for exp_num, acc_data in test_accuracies_step.items():
        plt.plot(acc_data['step'], acc_data['test/acc'], marker='o', label=f'Experiment {exp_num}')
    
    plt.title('Test Accuracy Across Experiments (Step-wise)')
    plt.xlabel('Step')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('test_accuracy_plot_step.png')

    # Generate test accuracy plot (epoch-wise)
    plt.figure(figsize=(10, 6))
    for exp_num, acc_data in test_accuracies_epoch.items():
        plt.plot(acc_data['epoch'], acc_data['test/acc'], marker='o', label=f'Experiment {exp_num}')
    
    plt.title('Test Accuracy Across Experiments (Epoch-wise)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('test_accuracy_plot_epoch.png')

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

    # Load and process optimization results
    opt_results_file = latest_run / "optimization_results.yaml"
    if opt_results_file.exists():
        opt_results = load_yaml(opt_results_file)
        
        # Create DataFrame for name and best parameters
        data = {'Parameter': ['name'] + list(opt_results['best_params'].keys()),
                'Value': [opt_results['name']] + list(opt_results['best_params'].values())}
        results_df = pd.DataFrame(data)

        # Add best value to the DataFrame
        results_df = pd.concat([results_df, pd.DataFrame([{'Parameter': 'best_value', 'Value': opt_results['best_value']}])], ignore_index=True)

        # Convert DataFrame to Markdown
        md_table = tabulate(results_df, headers='keys', tablefmt='pipe', floatfmt='.6f')

        # Save Markdown table to file
        with open('optimization_results.md', 'w') as f:
            f.write(md_table)

        print("Optimization results saved to 'optimization_results.md'")

if __name__ == "__main__":
    generate_results()
