"""
Script to analyze scaling law experiments and generate comprehensive plots.
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def power_law(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """Power law function: y = a * x^b"""
    return a * np.power(x, b)


def log_power_law(log_x: np.ndarray, log_a: float, b: float) -> np.ndarray:
    """Log-space power law: log(y) = log(a) + b * log(x)"""
    return log_a + b * log_x


def collect_experiment_results(checkpoint_dirs: List[str]) -> pd.DataFrame:
    """
    Collect results from multiple experiment directories.

    Args:
        checkpoint_dirs: List of directories containing experiment results

    Returns:
        DataFrame with consolidated results
    """
    all_results = []

    for checkpoint_dir in checkpoint_dirs:
        checkpoint_path = Path(checkpoint_dir)
        print(f"Collecting results from: {checkpoint_path}")

        # Look for sweep summary
        summary_file = checkpoint_path / "sweep_summary.json"
        if summary_file.exists():
            with open(summary_file, 'r') as f:
                summary = json.load(f)

            for result in summary['results']:
                if result['success']:
                    exp_data = result['results']
                    config = result['config']

                    # Extract key information
                    row = {
                        'experiment_name': result['exp_name'],
                        'checkpoint_dir': str(checkpoint_path),
                        'final_val_loss': exp_data.get('final_val_loss'),
                        'final_val_perplexity': exp_data.get('final_val_perplexity'),
                        'best_val_loss': exp_data.get('best_val_loss'),
                        'model_parameters': exp_data.get('model_parameters'),
                        'total_steps': exp_data.get('total_steps'),
                        # Model configuration
                        'd_model': config['model'].get('d_model'),
                        'n_layers': config['model'].get('n_layers'),
                        'n_heads': config['model'].get('n_heads'),
                        'max_seq_len': config['model'].get('max_seq_len'),
                        # Data configuration
                        'data_fraction': config['data'].get('data_fraction'),
                        'tokenizer_type': config['data'].get('tokenizer_type'),
                        'batch_size': config['data'].get('batch_size'),
                        # Training configuration
                        'learning_rate': config['training'].get('learning_rate'),
                        'num_epochs': config['training'].get('num_epochs'),
                        # Experiment metadata
                        'experiment_type': config['experiment'].get('name'),
                        'seed': config['data'].get('seed', 42)
                    }

                    all_results.append(row)

        else:
            # Look for individual experiment results
            for exp_dir in checkpoint_path.iterdir():
                if exp_dir.is_dir():
                    results_file = exp_dir / "results.json"
                    config_file = exp_dir / "config.yaml"

                    if results_file.exists() and config_file.exists():
                        with open(results_file, 'r') as f:
                            exp_data = json.load(f)

                        import yaml
                        with open(config_file, 'r') as f:
                            config = yaml.safe_load(f)

                        row = {
                            'experiment_name': exp_dir.name,
                            'checkpoint_dir': str(checkpoint_path),
                            'final_val_loss': exp_data.get('final_val_loss'),
                            'final_val_perplexity': exp_data.get('final_val_perplexity'),
                            'best_val_loss': exp_data.get('best_val_loss'),
                            'model_parameters': exp_data.get('model_parameters'),
                            'total_steps': exp_data.get('total_steps'),
                            'd_model': config['model'].get('d_model'),
                            'n_layers': config['model'].get('n_layers'),
                            'n_heads': config['model'].get('n_heads'),
                            'max_seq_len': config['model'].get('max_seq_len'),
                            'data_fraction': config['data'].get('data_fraction'),
                            'tokenizer_type': config['data'].get('tokenizer_type'),
                            'batch_size': config['data'].get('batch_size'),
                            'learning_rate': config['training'].get('learning_rate'),
                            'num_epochs': config['training'].get('num_epochs'),
                            'experiment_type': config['experiment'].get('name'),
                            'seed': config['data'].get('seed', 42)
                        }

                        all_results.append(row)

    if not all_results:
        raise ValueError("No experiment results found in provided directories")

    df = pd.DataFrame(all_results)
    print(f"Collected {len(df)} experiment results")
    return df


def fit_scaling_law(x: np.ndarray, y: np.ndarray, law_type: str = "power") -> Dict[str, Any]:
    """
    Fit scaling law to data.

    Args:
        x: Independent variable (e.g., model parameters)
        y: Dependent variable (e.g., loss)
        law_type: Type of law to fit ("power" or "log_power")

    Returns:
        Dictionary with fit parameters and statistics
    """
    # Remove any invalid data points
    valid_mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x_clean = x[valid_mask]
    y_clean = y[valid_mask]

    if len(x_clean) < 3:
        return {'success': False, 'error': 'Insufficient valid data points'}

    try:
        if law_type == "power":
            # Fit power law: y = a * x^b
            popt, pcov = curve_fit(power_law, x_clean, y_clean, p0=[1.0, -0.1])
            a, b = popt

            # Compute fit quality
            y_pred = power_law(x_clean, a, b)
            r_squared = 1 - np.sum((y_clean - y_pred) ** 2) / np.sum((y_clean - np.mean(y_clean)) ** 2)

            return {
                'success': True,
                'law_type': law_type,
                'parameters': {'a': a, 'b': b},
                'equation': f'y = {a:.3e} * x^{b:.3f}',
                'r_squared': r_squared,
                'fit_function': lambda x: power_law(x, a, b)
            }

        elif law_type == "log_power":
            # Fit in log space: log(y) = log(a) + b * log(x)
            log_x = np.log(x_clean)
            log_y = np.log(y_clean)

            popt, pcov = curve_fit(log_power_law, log_x, log_y, p0=[0.0, -0.1])
            log_a, b = popt
            a = np.exp(log_a)

            y_pred = power_law(x_clean, a, b)
            r_squared = 1 - np.sum((y_clean - y_pred) ** 2) / np.sum((y_clean - np.mean(y_clean)) ** 2)

            return {
                'success': True,
                'law_type': law_type,
                'parameters': {'a': a, 'b': b},
                'equation': f'y = {a:.3e} * x^{b:.3f}',
                'r_squared': r_squared,
                'fit_function': lambda x: power_law(x, a, b)
            }

    except Exception as e:
        return {'success': False, 'error': str(e)}


def analyze_scaling_dimensions(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze scaling across different dimensions.

    Args:
        df: DataFrame with experiment results

    Returns:
        Dictionary with scaling analysis results
    """
    scaling_results = {}

    # Group by different scaling dimensions
    scaling_dimensions = [
        ('d_model', 'Model Width'),
        ('n_layers', 'Model Depth'),
        ('max_seq_len', 'Context Length'),
        ('data_fraction', 'Data Size'),
        ('model_parameters', 'Total Parameters')
    ]

    for dim, title in scaling_dimensions:
        if dim in df.columns and df[dim].nunique() > 1:
            print(f"\nAnalyzing scaling with {title}...")

            # Group by dimension and compute statistics
            grouped = df.groupby(dim).agg({
                'final_val_loss': ['mean', 'std', 'count'],
                'best_val_loss': ['mean', 'std'],
                'model_parameters': 'first'
            }).reset_index()

            # Flatten column names
            grouped.columns = [f"{col[0]}_{col[1]}" if col[1] != '' else col[0] for col in grouped.columns]

            # Fit scaling law
            x_values = grouped[dim].values
            y_values = grouped['final_val_loss_mean'].values

            # Try both power law and log-power law fits
            power_fit = fit_scaling_law(x_values, y_values, "power")
            log_power_fit = fit_scaling_law(x_values, y_values, "log_power")

            # Choose best fit
            best_fit = power_fit
            if log_power_fit['success'] and power_fit['success']:
                if log_power_fit['r_squared'] > power_fit['r_squared']:
                    best_fit = log_power_fit

            scaling_results[dim] = {
                'title': title,
                'data': grouped,
                'power_fit': power_fit,
                'log_power_fit': log_power_fit,
                'best_fit': best_fit,
                'x_values': x_values,
                'y_values': y_values,
                'y_errors': grouped['final_val_loss_std'].values
            }

            if best_fit['success']:
                print(f"  Best fit: {best_fit['equation']} (R² = {best_fit['r_squared']:.3f})")
            else:
                print(f"  Fitting failed: {best_fit.get('error', 'Unknown error')}")

    return scaling_results


def create_scaling_plots(scaling_results: Dict[str, Any], output_dir: Path) -> None:
    """
    Create comprehensive scaling law plots.

    Args:
        scaling_results: Results from analyze_scaling_dimensions
        output_dir: Directory to save plots
    """
    # Create main scaling laws figure
    n_plots = len(scaling_results)
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    if n_plots == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    colors = sns.color_palette("husl", n_plots)

    for i, (dim, results) in enumerate(scaling_results.items()):
        ax = axes[i]

        x_vals = results['x_values']
        y_vals = results['y_values']
        y_errs = results['y_errors']

        # Plot data points with error bars
        ax.errorbar(x_vals, y_vals, yerr=y_errs, fmt='o', capsize=5,
                   color=colors[i], label='Data', markersize=8, alpha=0.8)

        # Plot best fit line
        if results['best_fit']['success']:
            x_fit = np.logspace(np.log10(x_vals.min()), np.log10(x_vals.max()), 100)
            y_fit = results['best_fit']['fit_function'](x_fit)

            ax.plot(x_fit, y_fit, '--', color=colors[i], alpha=0.8, linewidth=2,
                   label=f"Fit: {results['best_fit']['equation']}")

        ax.set_xlabel(results['title'])
        ax.set_ylabel('Validation Loss')
        ax.set_title(f"Scaling with {results['title']}")
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend()

    # Hide empty subplots
    for i in range(n_plots, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_dir / "scaling_laws.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Create individual detailed plots
    for dim, results in scaling_results.items():
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        x_vals = results['x_values']
        y_vals = results['y_values']
        y_errs = results['y_errors']

        # Plot 1: Log-log scaling plot
        ax1.errorbar(x_vals, y_vals, yerr=y_errs, fmt='o', capsize=5,
                    markersize=10, label='Experimental Data')

        if results['best_fit']['success']:
            x_fit = np.logspace(np.log10(x_vals.min()), np.log10(x_vals.max()), 100)
            y_fit = results['best_fit']['fit_function'](x_fit)
            ax1.plot(x_fit, y_fit, '--', linewidth=3, alpha=0.8,
                    label=f"Best Fit: {results['best_fit']['equation']}")

        ax1.set_xlabel(results['title'])
        ax1.set_ylabel('Validation Loss')
        ax1.set_title(f"Scaling Law: {results['title']}")
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Plot 2: Residuals
        if results['best_fit']['success']:
            y_pred = results['best_fit']['fit_function'](x_vals)
            residuals = (y_vals - y_pred) / y_vals  # Relative residuals

            ax2.scatter(x_vals, residuals, s=100, alpha=0.7)
            ax2.axhline(y=0, color='red', linestyle='--', alpha=0.7)
            ax2.set_xlabel(results['title'])
            ax2.set_ylabel('Relative Residuals')
            ax2.set_title('Fit Quality: Relative Residuals')
            ax2.set_xscale('log')
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / f"scaling_{dim}_detailed.png", dpi=300, bbox_inches='tight')
        plt.close()


def create_comparison_plots(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Create comparison plots across different experiment types.

    Args:
        df: DataFrame with experiment results
        output_dir: Directory to save plots
    """
    # Plot 1: Loss vs Parameters colored by experiment type
    plt.figure(figsize=(12, 8))

    experiment_types = df['experiment_type'].unique()
    colors = sns.color_palette("Set2", len(experiment_types))

    for i, exp_type in enumerate(experiment_types):
        exp_data = df[df['experiment_type'] == exp_type]
        plt.scatter(exp_data['model_parameters'], exp_data['final_val_loss'],
                   label=exp_type, s=100, alpha=0.7, color=colors[i])

    plt.xlabel('Model Parameters')
    plt.ylabel('Final Validation Loss')
    plt.title('Model Performance vs Parameters')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "performance_vs_parameters.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 2: Training efficiency (loss vs computational budget)
    if 'total_steps' in df.columns:
        plt.figure(figsize=(12, 8))

        # Estimate compute as parameters * steps
        df['compute_budget'] = df['model_parameters'] * df['total_steps']

        for i, exp_type in enumerate(experiment_types):
            exp_data = df[df['experiment_type'] == exp_type]
            plt.scatter(exp_data['compute_budget'], exp_data['final_val_loss'],
                       label=exp_type, s=100, alpha=0.7, color=colors[i])

        plt.xlabel('Compute Budget (Parameters × Steps)')
        plt.ylabel('Final Validation Loss')
        plt.title('Training Efficiency: Loss vs Compute')
        plt.xscale('log')
        plt.yscale('log')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "efficiency_vs_compute.png", dpi=300, bbox_inches='tight')
        plt.close()

    # Plot 3: Ablation studies comparison
    ablation_experiments = [exp for exp in experiment_types if 'ablation' in exp.lower()]

    if ablation_experiments:
        fig, axes = plt.subplots(1, len(ablation_experiments), figsize=(6 * len(ablation_experiments), 6))
        if len(ablation_experiments) == 1:
            axes = [axes]

        for i, exp_type in enumerate(ablation_experiments):
            exp_data = df[df['experiment_type'] == exp_type]

            # Determine ablation variable
            if 'tokenizer' in exp_type.lower():
                var = 'tokenizer_type'
                title = 'Tokenizer Comparison'
            elif 'optimizer' in exp_type.lower():
                var = 'learning_rate'
                title = 'Learning Rate Comparison'
            else:
                continue

            if var in exp_data.columns:
                grouped = exp_data.groupby(var)['final_val_loss'].agg(['mean', 'std']).reset_index()

                axes[i].bar(range(len(grouped)), grouped['mean'],
                           yerr=grouped['std'], capsize=5, alpha=0.7)
                axes[i].set_xticks(range(len(grouped)))
                axes[i].set_xticklabels(grouped[var], rotation=45)
                axes[i].set_ylabel('Final Validation Loss')
                axes[i].set_title(title)
                axes[i].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / "ablation_comparisons.png", dpi=300, bbox_inches='tight')
        plt.close()


def generate_summary_table(scaling_results: Dict[str, Any], output_dir: Path) -> None:
    """
    Generate summary table of scaling law results.

    Args:
        scaling_results: Results from analyze_scaling_dimensions
        output_dir: Directory to save table
    """
    summary_data = []

    for dim, results in scaling_results.items():
        if results['best_fit']['success']:
            fit = results['best_fit']
            summary_data.append({
                'Scaling Dimension': results['title'],
                'Scaling Exponent (b)': fit['parameters']['b'],
                'Coefficient (a)': fit['parameters']['a'],
                'R²': fit['r_squared'],
                'Equation': fit['equation'],
                'Data Points': len(results['x_values'])
            })

    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.round({'Scaling Exponent (b)': 3, 'R²': 3})

        # Save as CSV
        summary_df.to_csv(output_dir / "scaling_laws_summary.csv", index=False)

        # Save as formatted table
        with open(output_dir / "scaling_laws_table.txt", 'w') as f:
            f.write("Scaling Laws Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(summary_df.to_string(index=False))
            f.write("\n\n")
            f.write("Notes:\n")
            f.write("- Scaling exponent (b): How loss scales with the dimension\n")
            f.write("- Negative values indicate loss decreases as dimension increases\n")
            f.write("- R² indicates quality of fit (closer to 1 is better)\n")

        print(f"Summary table saved to {output_dir / 'scaling_laws_summary.csv'}")


def main():
    parser = argparse.ArgumentParser(description="Analyze scaling law experiments")
    parser.add_argument("--checkpoint_dirs", type=str, required=True,
                       help="Comma-separated list of checkpoint directories")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for analysis results")

    args = parser.parse_args()

    # Parse checkpoint directories
    checkpoint_dirs = [d.strip() for d in args.checkpoint_dirs.split(',')]

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Analyzing scaling law experiments...")

    # Collect results
    df = collect_experiment_results(checkpoint_dirs)

    # Save consolidated results
    df.to_csv(output_dir / "all_experiment_results.csv", index=False)
    print(f"Consolidated results saved to {output_dir / 'all_experiment_results.csv'}")

    # Analyze scaling laws
    scaling_results = analyze_scaling_dimensions(df)

    # Create plots
    print("Generating plots...")
    create_scaling_plots(scaling_results, output_dir)
    create_comparison_plots(df, output_dir)

    # Generate summary table
    generate_summary_table(scaling_results, output_dir)

    # Save detailed results
    results_summary = {
        'total_experiments': len(df),
        'experiment_types': df['experiment_type'].unique().tolist(),
        'scaling_results': {
            dim: {
                'title': results['title'],
                'best_fit': results['best_fit'] if results['best_fit']['success'] else None,
                'num_points': len(results['x_values'])
            }
            for dim, results in scaling_results.items()
        }
    }

    with open(output_dir / "analysis_summary.json", 'w') as f:
        json.dump(results_summary, f, indent=2)

    print(f"\nAnalysis completed! Results saved in: {output_dir}")
    print(f"Generated plots:")
    print(f"  - scaling_laws.png: Overview of all scaling laws")
    print(f"  - scaling_*_detailed.png: Detailed plots for each dimension")
    print(f"  - performance_vs_parameters.png: Performance comparison")
    print(f"  - efficiency_vs_compute.png: Training efficiency analysis")


if __name__ == "__main__":
    main()