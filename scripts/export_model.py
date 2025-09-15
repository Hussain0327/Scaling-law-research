"""
Script to export trained models for inference and deployment.
"""

import os
import sys
import json
import argparse
import torch
from pathlib import Path
from typing import Dict, Any, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models.tiny_gpt import TinyGPT
from data.tokenizers import CharacterTokenizer, SubwordTokenizer


def find_best_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """
    Find the best checkpoint in a directory based on validation loss.

    Args:
        checkpoint_dir: Directory containing checkpoints

    Returns:
        Path to best checkpoint or None if not found
    """
    checkpoint_path = Path(checkpoint_dir)

    # Look for best_model.pt first
    best_model_path = checkpoint_path / "best_model.pt"
    if best_model_path.exists():
        return str(best_model_path)

    # Look for final_model.pt
    final_model_path = checkpoint_path / "final_model.pt"
    if final_model_path.exists():
        return str(final_model_path)

    # Look for checkpoint files and find the best one
    checkpoint_files = list(checkpoint_path.glob("checkpoint_step_*.pt"))
    if not checkpoint_files:
        return None

    best_checkpoint = None
    best_loss = float('inf')

    for checkpoint_file in checkpoint_files:
        try:
            checkpoint = torch.load(checkpoint_file, map_location='cpu')
            val_loss = checkpoint.get('best_val_loss', float('inf'))

            if val_loss < best_loss:
                best_loss = val_loss
                best_checkpoint = str(checkpoint_file)
        except Exception as e:
            print(f"Warning: Could not load {checkpoint_file}: {e}")
            continue

    return best_checkpoint


def export_model(
    checkpoint_path: str,
    output_dir: str,
    export_format: str = "pytorch",
    include_tokenizer: bool = True,
    optimize_for_inference: bool = True
) -> Dict[str, Any]:
    """
    Export a trained model for inference.

    Args:
        checkpoint_path: Path to model checkpoint
        output_dir: Directory to save exported model
        export_format: Export format ('pytorch', 'onnx', 'torchscript')
        include_tokenizer: Whether to include tokenizer
        optimize_for_inference: Whether to optimize model for inference

    Returns:
        Export summary
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint from: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    config = checkpoint['config']
    model_config = config['model']

    # Create and load model
    model = TinyGPT(**model_config)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimize_for_inference:
        model.eval()
        # Disable dropout for inference
        for module in model.modules():
            if hasattr(module, 'dropout'):
                module.dropout.p = 0.0

    print(f"Loaded model with {model.count_parameters():,} parameters")

    export_info = {
        'model_config': model_config,
        'training_config': config.get('training', {}),
        'model_parameters': model.count_parameters(),
        'final_val_loss': checkpoint.get('best_val_loss'),
        'total_steps': checkpoint.get('step'),
        'export_format': export_format,
        'optimized_for_inference': optimize_for_inference
    }

    # Export model in specified format
    if export_format == "pytorch":
        # Save as standard PyTorch model
        model_path = output_path / "model.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_config': model_config,
            'export_info': export_info
        }, model_path)

        print(f"PyTorch model saved to: {model_path}")

    elif export_format == "torchscript":
        # Export as TorchScript for deployment
        model.eval()

        # Create example input
        example_input = torch.randint(0, model_config['vocab_size'], (1, 10))

        try:
            # Try tracing first
            traced_model = torch.jit.trace(model, example_input)
            script_path = output_path / "model_traced.pt"
            traced_model.save(str(script_path))
            print(f"TorchScript traced model saved to: {script_path}")

        except Exception as e:
            print(f"Tracing failed: {e}, trying scripting...")
            try:
                scripted_model = torch.jit.script(model)
                script_path = output_path / "model_scripted.pt"
                scripted_model.save(str(script_path))
                print(f"TorchScript scripted model saved to: {script_path}")
            except Exception as e:
                print(f"Scripting also failed: {e}")

    elif export_format == "onnx":
        # Export as ONNX for cross-platform deployment
        try:
            import torch.onnx

            model.eval()
            example_input = torch.randint(0, model_config['vocab_size'], (1, 10))

            onnx_path = output_path / "model.onnx"
            torch.onnx.export(
                model,
                example_input,
                str(onnx_path),
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input_ids'],
                output_names=['logits'],
                dynamic_axes={
                    'input_ids': {0: 'batch_size', 1: 'sequence_length'},
                    'logits': {0: 'batch_size', 1: 'sequence_length'}
                }
            )
            print(f"ONNX model saved to: {onnx_path}")

        except ImportError:
            print("ONNX export requires torch.onnx. Skipping ONNX export.")
        except Exception as e:
            print(f"ONNX export failed: {e}")

    # Save model configuration and metadata
    config_path = output_path / "model_config.json"
    with open(config_path, 'w') as f:
        json.dump(export_info, f, indent=2)

    # Export tokenizer if requested
    if include_tokenizer:
        try:
            # Try to recreate tokenizer from config
            data_config = config.get('data', {})
            tokenizer_type = data_config.get('tokenizer_type', 'char')

            if tokenizer_type == 'char':
                # For character tokenizer, we need the vocabulary
                # This is a limitation - we'd need to save the tokenizer during training
                print("Warning: Character tokenizer vocabulary not available in checkpoint.")
                print("Consider saving tokenizer during training for complete export.")

            elif tokenizer_type == 'subword':
                tokenizer = SubwordTokenizer(data_config.get('model_name', 'gpt2'))

                # Save tokenizer info
                tokenizer_info = {
                    'type': 'subword',
                    'model_name': data_config.get('model_name', 'gpt2'),
                    'vocab_size': tokenizer.vocab_size,
                    'pad_token_id': tokenizer.pad_token_id,
                    'eos_token_id': tokenizer.eos_token_id,
                    'bos_token_id': tokenizer.bos_token_id
                }

                tokenizer_path = output_path / "tokenizer_info.json"
                with open(tokenizer_path, 'w') as f:
                    json.dump(tokenizer_info, f, indent=2)

                print(f"Tokenizer info saved to: {tokenizer_path}")

        except Exception as e:
            print(f"Warning: Could not export tokenizer: {e}")

    # Create inference script template
    inference_script = f'''"""
Inference script for exported TinyGPT model.
Generated automatically during model export.
"""

import torch
import json
from pathlib import Path

def load_model(model_dir: str):
    """Load exported model and configuration."""
    model_dir = Path(model_dir)

    # Load configuration
    with open(model_dir / "model_config.json", 'r') as f:
        export_info = json.load(f)

    model_config = export_info['model_config']

    # Load model (adjust import path as needed)
    from models.tiny_gpt import TinyGPT
    model = TinyGPT(**model_config)

    # Load state dict
    checkpoint = torch.load(model_dir / "model.pt", map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, export_info

def generate_text(model, input_ids, max_new_tokens=50, temperature=0.8):
    """Generate text using the model."""
    model.eval()
    with torch.no_grad():
        generated = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True
        )
    return generated

# Example usage:
if __name__ == "__main__":
    # Load model
    model, info = load_model(".")

    print(f"Model loaded with {{info['model_parameters']:,}} parameters")
    print(f"Final validation loss: {{info.get('final_val_loss', 'N/A')}}")

    # Example generation (you'll need to tokenize input appropriately)
    input_ids = torch.randint(0, info['model_config']['vocab_size'], (1, 5))
    output = generate_text(model, input_ids, max_new_tokens=20)

    print(f"Generated sequence shape: {{output.shape}}")
'''

    script_path = output_path / "inference.py"
    with open(script_path, 'w') as f:
        f.write(inference_script)

    print(f"Inference script saved to: {script_path}")

    # Save export summary
    summary_path = output_path / "export_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(export_info, f, indent=2)

    print(f"Export completed! Summary saved to: {summary_path}")

    return export_info


def main():
    parser = argparse.ArgumentParser(description="Export trained TinyGPT model")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                       help="Directory containing model checkpoint")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for exported model")
    parser.add_argument("--checkpoint_file", type=str, default=None,
                       help="Specific checkpoint file (if not provided, best will be found)")
    parser.add_argument("--format", type=str, choices=["pytorch", "torchscript", "onnx"],
                       default="pytorch", help="Export format")
    parser.add_argument("--include_tokenizer", action="store_true",
                       help="Include tokenizer information")
    parser.add_argument("--optimize", action="store_true",
                       help="Optimize model for inference")

    args = parser.parse_args()

    # Find checkpoint file
    if args.checkpoint_file:
        checkpoint_path = args.checkpoint_file
    else:
        checkpoint_path = find_best_checkpoint(args.checkpoint_dir)

    if not checkpoint_path:
        print(f"No suitable checkpoint found in {args.checkpoint_dir}")
        return

    print(f"Using checkpoint: {checkpoint_path}")

    # Export model
    export_info = export_model(
        checkpoint_path=checkpoint_path,
        output_dir=args.output_dir,
        export_format=args.format,
        include_tokenizer=args.include_tokenizer,
        optimize_for_inference=args.optimize
    )

    print("\nExport completed successfully!")
    print(f"Model parameters: {export_info['model_parameters']:,}")
    print(f"Export format: {export_info['export_format']}")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()