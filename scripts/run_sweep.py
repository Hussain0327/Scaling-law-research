"""
Script to run parameter sweeps for scaling law experiments.
Supports sweeping across different model configurations.
"""

import os
import sys
import json
import argparse
import itertools
from pathlib import Path
from typing import Dict, List, Any
import yaml
import subprocess

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from train import main as train_main


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def generate_sweep_configs(base_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Generate configurations for parameter sweep based on sweep specification.

    Args:
        base_config: Base configuration with sweep parameters

    Returns:
        List of configurations for each sweep combination
    """
    if 'sweep' not in base_config:
        return [base_config]

    sweep_config = base_config['sweep']
    parameter = sweep_config['parameter']
    values = sweep_config['values']

    configs = []

    for value in values:
        config = base_config.copy()

        # Remove sweep section from config
        del config['sweep']

        # Update the swept parameter
        if '.' in parameter:
            # Handle nested parameters like 'model.d_model'
            keys = parameter.split('.')
            target = config
            for key in keys[:-1]:
                target = target[key]
            target[keys[-1]] = value
        else:
            config[parameter] = value

        # Handle special adjustments
        if sweep_config.get('adjust_heads', False) and parameter == 'd_model':
            # Adjust number of heads to be divisible by d_model
            if value >= 64:
                config['model']['n_heads'] = 8
            elif value >= 32:
                config['model']['n_heads'] = 4
            else:
                config['model']['n_heads'] = 2

        if sweep_config.get('adjust_data_length', False) and parameter == 'max_seq_len':
            # Adjust data max_length to match model max_seq_len
            config['data']['max_length'] = value
            config['data']['stride'] = value // 2

        # Handle secondary parameter sweeps
        if 'secondary_parameter' in sweep_config:
            secondary_param = sweep_config['secondary_parameter']
            secondary_values = sweep_config['secondary_values']

            # Create configs for each combination of primary and secondary parameters
            secondary_configs = []
            for secondary_value in secondary_values:
                secondary_config = config.copy()

                if '.' in secondary_param:
                    keys = secondary_param.split('.')
                    target = secondary_config
                    for key in keys[:-1]:
                        target = target[key]
                    target[keys[-1]] = secondary_value
                else:
                    secondary_config[secondary_param] = secondary_value

                # Update experiment name to include both parameters
                secondary_config['experiment']['name'] = f"{config['experiment']['name']}_{parameter}_{value}_{secondary_param}_{secondary_value}"
                secondary_configs.append(secondary_config)

            configs.extend(secondary_configs)
        else:
            # Update experiment name to include swept parameter
            config['experiment']['name'] = f"{config['experiment']['name']}_{parameter}_{value}"
            configs.append(config)

    return configs


def run_single_experiment(config: Dict[str, Any], output_dir: str, seed: int = None) -> Dict[str, Any]:
    """
    Run a single experiment with given configuration.

    Args:
        config: Experiment configuration
        output_dir: Directory to save results
        seed: Random seed for experiment

    Returns:
        Experiment results
    """
    # Create experiment-specific output directory
    exp_name = config['experiment']['name']
    if seed is not None:
        exp_name = f"{exp_name}_seed_{seed}"
        config['data']['seed'] = seed

    exp_dir = Path(output_dir) / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Save configuration
    config_path = exp_dir / "config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f, indent=2)

    print(f"Running experiment: {exp_name}")
    print(f"Output directory: {exp_dir}")

    try:
        # Run training
        import tempfile
        import sys
        from train import setup_experiment, Trainer

        # Setup experiment
        model, train_loader, val_loader = setup_experiment(config)

        # Create trainer
        trainer = Trainer(
            model=model,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            config=config,
            save_dir=str(exp_dir),
            use_wandb=False  # Disable wandb for sweeps
        )

        # Train model
        results = trainer.train()

        # Save results
        results_path = exp_dir / "results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Experiment {exp_name} completed successfully")
        print(f"Final validation loss: {results['final_val_loss']:.4f}")

        return {
            'exp_name': exp_name,
            'success': True,
            'results': results,
            'config': config
        }

    except Exception as e:
        print(f"Experiment {exp_name} failed with error: {e}")
        return {
            'exp_name': exp_name,
            'success': False,
            'error': str(e),
            'config': config
        }


def run_sweep(config_path: str, output_dir: str, seeds: List[int] = None, max_parallel: int = 1) -> Dict[str, Any]:
    """
    Run parameter sweep experiments.

    Args:
        config_path: Path to sweep configuration file
        output_dir: Directory to save all experiment results
        seeds: List of random seeds to run for each configuration
        max_parallel: Maximum number of parallel experiments (future feature)

    Returns:
        Summary of all experiments
    """
    # Load base configuration
    base_config = load_config(config_path)

    # Generate sweep configurations
    sweep_configs = generate_sweep_configs(base_config)

    print(f"Generated {len(sweep_configs)} configurations for sweep")

    if seeds is None:
        seeds = [42]  # Default seed

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Run experiments
    all_results = []
    total_experiments = len(sweep_configs) * len(seeds)

    print(f"Running {total_experiments} total experiments...")

    for i, config in enumerate(sweep_configs):
        print(f"\nConfiguration {i+1}/{len(sweep_configs)}")

        for j, seed in enumerate(seeds):
            print(f"Seed {j+1}/{len(seeds)}: {seed}")

            result = run_single_experiment(config, output_dir, seed)
            all_results.append(result)

    # Save sweep summary
    summary = {
        'sweep_config': base_config,
        'total_experiments': total_experiments,
        'successful_experiments': sum(1 for r in all_results if r['success']),
        'failed_experiments': sum(1 for r in all_results if not r['success']),
        'results': all_results
    }

    summary_path = output_path / "sweep_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nSweep completed!")
    print(f"Successful experiments: {summary['successful_experiments']}/{total_experiments}")
    print(f"Failed experiments: {summary['failed_experiments']}/{total_experiments}")
    print(f"Summary saved to: {summary_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Run parameter sweep experiments")
    parser.add_argument("--config", type=str, required=True, help="Path to sweep configuration file")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for experiments")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42], help="Random seeds for experiments")
    parser.add_argument("--max_parallel", type=int, default=1, help="Maximum parallel experiments")
    parser.add_argument("--dry_run", action="store_true", help="Show configurations without running")

    args = parser.parse_args()

    if args.dry_run:
        # Show what configurations would be generated
        base_config = load_config(args.config)
        configs = generate_sweep_configs(base_config)

        print(f"Would generate {len(configs)} configurations:")
        for i, config in enumerate(configs):
            print(f"{i+1}. {config['experiment']['name']}")
            if 'model' in config:
                print(f"   Model: d_model={config['model'].get('d_model', 'N/A')}, "
                      f"n_layers={config['model'].get('n_layers', 'N/A')}, "
                      f"n_heads={config['model'].get('n_heads', 'N/A')}")
        print(f"\nWith seeds {args.seeds}, total experiments: {len(configs) * len(args.seeds)}")
        return

    # Run the sweep
    summary = run_sweep(
        config_path=args.config,
        output_dir=args.output_dir,
        seeds=args.seeds,
        max_parallel=args.max_parallel
    )

    # Print final summary
    if summary['failed_experiments'] > 0:
        print("\nFailed experiments:")
        for result in summary['results']:
            if not result['success']:
                print(f"- {result['exp_name']}: {result['error']}")

    print(f"\nSweep results saved in: {args.output_dir}")


if __name__ == "__main__":
    main()