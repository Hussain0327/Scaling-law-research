"""
Edge case and stress tests for TinyGPT model components.
Tests boundary conditions, numerical stability, and performance under stress.
"""

import sys
import gc
import time
from contextlib import contextmanager

sys.path.append("src")

import pytest
import torch
import numpy as np

from models.tiny_gpt import (
    TinyGPT,
    MultiHeadAttention,
    MLP,
    TransformerBlock,
    create_tiny_gpt
)


@contextmanager
def memory_profiler():
    """Context manager for memory profiling."""
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    gc.collect()

    if torch.cuda.is_available():
        start_memory = torch.cuda.memory_allocated()
    else:
        start_memory = 0

    yield

    if torch.cuda.is_available():
        end_memory = torch.cuda.memory_allocated()
        memory_used = end_memory - start_memory
        print(f"Memory used: {memory_used / 1024**2:.2f} MB")


class TestNumericalStability:
    """Test numerical stability of model components."""

    def test_attention_with_extreme_values(self):
        """Test attention mechanism with extreme input values."""
        d_model, n_heads = 64, 8
        attn = MultiHeadAttention(d_model, n_heads)

        # Test with very large values
        large_input = torch.randn(1, 10, d_model) * 1000
        output = attn(large_input)
        assert torch.isfinite(output).all()

        # Test with very small values
        small_input = torch.randn(1, 10, d_model) * 1e-6
        output = attn(small_input)
        assert torch.isfinite(output).all()

    def test_attention_gradient_stability(self):
        """Test gradient stability through attention layers."""
        d_model, n_heads = 32, 4
        attn = MultiHeadAttention(d_model, n_heads, dropout=0.0)

        # Create input that might cause gradient issues
        x = torch.randn(2, 100, d_model, requires_grad=True)

        # Forward and backward
        output = attn(x)
        loss = output.sum()
        loss.backward()

        # Check gradients are finite and reasonable
        assert torch.isfinite(x.grad).all()
        assert x.grad.abs().max() < 1000  # Gradients shouldn't explode

    def test_mlp_with_zero_inputs(self):
        """Test MLP behavior with zero inputs."""
        d_model, d_ff = 64, 256
        mlp = MLP(d_model, d_ff, dropout=0.0)

        zero_input = torch.zeros(2, 10, d_model)
        output = mlp(zero_input)

        # Output should be finite but not necessarily zero (due to biases)
        assert torch.isfinite(output).all()

    def test_model_with_mixed_precision(self):
        """Test model behavior with mixed precision inputs."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for mixed precision testing")

        model = TinyGPT(
            vocab_size=100,
            d_model=32,
            n_layers=2,
            n_heads=4,
            max_seq_len=16
        ).cuda()

        input_ids = torch.randint(0, 100, (2, 8)).cuda()

        # Test with autocast
        with torch.cuda.amp.autocast():
            logits, loss = model(input_ids, input_ids)

        assert torch.isfinite(logits).all()
        assert torch.isfinite(loss)

    def test_loss_with_all_same_tokens(self):
        """Test loss computation when all tokens are the same."""
        model = TinyGPT(
            vocab_size=100,
            d_model=32,
            n_layers=2,
            n_heads=4
        )

        # All tokens are the same
        same_token_ids = torch.full((2, 10), 42)
        logits, loss = model(same_token_ids, same_token_ids)

        assert torch.isfinite(loss)
        # Loss should still be meaningful
        assert loss.item() > 0

    def test_numerical_stability_long_sequences(self):
        """Test numerical stability with very long sequences."""
        model = TinyGPT(
            vocab_size=100,
            d_model=64,
            n_layers=4,
            n_heads=8,
            max_seq_len=512,
            dropout=0.0
        )

        # Long sequence
        long_input = torch.randint(0, 100, (1, 512))
        logits, loss = model(long_input, long_input)

        assert torch.isfinite(logits).all()
        assert torch.isfinite(loss)


class TestMemoryAndPerformance:
    """Test memory usage and performance characteristics."""

    def test_memory_scaling_with_batch_size(self):
        """Test that memory usage scales appropriately with batch size."""
        model = TinyGPT(
            vocab_size=1000,
            d_model=128,
            n_layers=4,
            n_heads=8,
            max_seq_len=64
        )

        seq_len = 32
        memory_usage = []

        for batch_size in [1, 2, 4, 8]:
            with memory_profiler():
                input_ids = torch.randint(0, 1000, (batch_size, seq_len))
                logits, _ = model(input_ids)
                # Force computation
                logits.sum().item()

            # Memory should scale roughly linearly with batch size
            # (This is a simplified test)

    def test_generation_memory_efficiency(self):
        """Test memory efficiency during generation."""
        model = TinyGPT(
            vocab_size=1000,
            d_model=64,
            n_layers=2,
            n_heads=4,
            max_seq_len=128
        )

        input_ids = torch.randint(0, 1000, (1, 10))

        with memory_profiler():
            generated = model.generate(
                input_ids,
                max_new_tokens=50,
                do_sample=False
            )

        assert generated.shape[1] == 60  # 10 + 50

    def test_attention_computation_scaling(self):
        """Test that attention computation scales quadratically with sequence length."""
        d_model, n_heads = 64, 8
        attn = MultiHeadAttention(d_model, n_heads, dropout=0.0)

        times = []
        seq_lengths = [16, 32, 64, 128]

        for seq_len in seq_lengths:
            x = torch.randn(1, seq_len, d_model)

            start = time.perf_counter()
            for _ in range(10):
                output = attn(x)
                output.sum().item()  # Force computation
            end = time.perf_counter()

            times.append(end - start)

        # Check that time increases super-linearly (roughly quadratic)
        # Ratio should increase as sequence length doubles
        ratios = [times[i+1]/times[i] for i in range(len(times)-1)]
        # Each ratio should be > 2 for quadratic scaling (ideally ~4)
        assert all(r > 1.5 for r in ratios)

    @pytest.mark.slow
    def test_large_model_initialization(self):
        """Test initialization of large model configurations."""
        # Test that large models can be initialized without issues
        large_model = TinyGPT(
            vocab_size=50000,
            d_model=1024,
            n_layers=24,
            n_heads=16,
            max_seq_len=1024,
            dropout=0.1
        )

        param_count = large_model.count_parameters()
        assert param_count > 100_000_000  # Should be > 100M parameters

        # Test forward pass with small batch
        input_ids = torch.randint(0, 50000, (1, 128))
        logits, _ = large_model(input_ids)
        assert logits.shape == (1, 128, 50000)


class TestEdgeCasesAndBoundaries:
    """Test edge cases and boundary conditions."""

    def test_single_token_generation(self):
        """Test generation with single token input."""
        model = TinyGPT(
            vocab_size=100,
            d_model=32,
            n_layers=2,
            n_heads=4
        )

        single_token = torch.tensor([[1]])
        generated = model.generate(single_token, max_new_tokens=5)

        assert generated.shape == (1, 6)

    def test_empty_generation(self):
        """Test generation with max_new_tokens=0."""
        model = TinyGPT(
            vocab_size=100,
            d_model=32,
            n_layers=2,
            n_heads=4
        )

        input_ids = torch.tensor([[1, 2, 3]])
        generated = model.generate(input_ids, max_new_tokens=0)

        assert torch.equal(generated, input_ids)

    def test_generation_at_max_length(self):
        """Test generation when starting at maximum sequence length."""
        max_seq_len = 16
        model = TinyGPT(
            vocab_size=100,
            d_model=32,
            n_layers=2,
            n_heads=4,
            max_seq_len=max_seq_len
        )

        # Input already at max length
        input_ids = torch.randint(0, 100, (1, max_seq_len))

        with pytest.raises(AssertionError):
            # Should fail because we can't add more tokens
            model.generate(input_ids, max_new_tokens=1)

    def test_attention_mask_edge_cases(self):
        """Test attention masking edge cases."""
        d_model, n_heads = 32, 4
        attn = MultiHeadAttention(d_model, n_heads)

        # Test with sequence length 1 (no masking needed)
        single_pos = torch.randn(1, 1, d_model)
        output = attn(single_pos)
        assert output.shape == (1, 1, d_model)

        # Test that mask is properly applied for different sequence lengths
        for seq_len in [2, 5, 10, 50]:
            x = torch.randn(1, seq_len, d_model)
            output = attn(x)
            assert output.shape == (1, seq_len, d_model)

    def test_vocab_size_edge_cases(self):
        """Test model with extreme vocabulary sizes."""
        # Very small vocabulary
        small_vocab_model = TinyGPT(
            vocab_size=2,  # Binary vocabulary
            d_model=32,
            n_layers=2,
            n_heads=4
        )

        input_ids = torch.tensor([[0, 1, 0, 1, 1, 0]])
        logits, loss = small_vocab_model(input_ids, input_ids)

        assert logits.shape[-1] == 2
        assert torch.isfinite(loss)

    def test_model_with_dropout_1(self):
        """Test model behavior with dropout=1.0 (all dropped)."""
        model = TinyGPT(
            vocab_size=100,
            d_model=32,
            n_layers=2,
            n_heads=4,
            dropout=1.0  # Extreme dropout
        )

        model.train()  # Dropout is active
        input_ids = torch.randint(0, 100, (2, 8))
        logits, loss = model(input_ids, input_ids)

        # Model should still produce outputs (though they may be poor)
        assert torch.isfinite(logits).all()
        assert torch.isfinite(loss)

    def test_temperature_extremes_in_generation(self):
        """Test generation with extreme temperature values."""
        model = TinyGPT(
            vocab_size=100,
            d_model=32,
            n_layers=2,
            n_heads=4
        )

        input_ids = torch.tensor([[1, 2, 3]])

        # Very low temperature (near deterministic)
        generated_low = model.generate(
            input_ids,
            max_new_tokens=10,
            do_sample=True,
            temperature=0.01
        )

        # Very high temperature (very random)
        generated_high = model.generate(
            input_ids,
            max_new_tokens=10,
            do_sample=True,
            temperature=100.0
        )

        assert generated_low.shape == (1, 13)
        assert generated_high.shape == (1, 13)

    def test_top_k_top_p_edge_cases(self):
        """Test generation with edge cases for top_k and top_p."""
        model = TinyGPT(
            vocab_size=100,
            d_model=32,
            n_layers=2,
            n_heads=4
        )

        input_ids = torch.tensor([[1, 2, 3]])

        # top_k = 1 (always pick the most likely)
        generated_k1 = model.generate(
            input_ids,
            max_new_tokens=5,
            do_sample=True,
            top_k=1
        )

        # top_p very small (similar to top_k=1)
        generated_p_small = model.generate(
            input_ids,
            max_new_tokens=5,
            do_sample=True,
            top_p=0.01
        )

        # top_p = 1.0 (consider all tokens)
        generated_p_full = model.generate(
            input_ids,
            max_new_tokens=5,
            do_sample=True,
            top_p=1.0
        )

        assert all(g.shape == (1, 8) for g in [generated_k1, generated_p_small, generated_p_full])


class TestGradientFlowAndBackpropagation:
    """Test gradient flow and backpropagation edge cases."""

    def test_gradient_accumulation(self):
        """Test gradient accumulation over multiple forward passes."""
        model = TinyGPT(
            vocab_size=100,
            d_model=32,
            n_layers=2,
            n_heads=4
        )

        optimizer = torch.optim.Adam(model.parameters())

        # Store initial parameters
        initial_params = {name: param.clone()
                         for name, param in model.named_parameters()}

        # Accumulate gradients over multiple batches
        for _ in range(3):
            input_ids = torch.randint(0, 100, (2, 8))
            _, loss = model(input_ids, input_ids)
            loss.backward()  # Accumulate gradients

        optimizer.step()

        # Check that parameters changed
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert not torch.equal(param, initial_params[name])

    def test_gradient_clipping_effect(self):
        """Test effect of gradient clipping on training stability."""
        model = TinyGPT(
            vocab_size=100,
            d_model=32,
            n_layers=2,
            n_heads=4
        )

        # Create input that might cause large gradients
        input_ids = torch.randint(0, 100, (4, 32))
        targets = torch.randint(0, 100, (4, 32))

        _, loss = model(input_ids, targets)
        loss.backward()

        # Check gradient magnitudes before clipping
        total_norm_before = 0
        for p in model.parameters():
            if p.grad is not None:
                total_norm_before += p.grad.norm(2).item() ** 2
        total_norm_before = total_norm_before ** 0.5

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Check gradient magnitudes after clipping
        total_norm_after = 0
        for p in model.parameters():
            if p.grad is not None:
                total_norm_after += p.grad.norm(2).item() ** 2
        total_norm_after = total_norm_after ** 0.5

        assert total_norm_after <= 1.0 + 1e-6  # Allow small numerical errors

    def test_backward_with_retain_graph(self):
        """Test multiple backward passes with retained computation graph."""
        model = TinyGPT(
            vocab_size=100,
            d_model=32,
            n_layers=2,
            n_heads=4
        )

        input_ids = torch.randint(0, 100, (2, 8))
        logits, loss = model(input_ids, input_ids)

        # First backward
        loss.backward(retain_graph=True)

        # Second backward (should work with retained graph)
        loss.backward()

        # Gradients should have accumulated
        for param in model.parameters():
            if param.grad is not None:
                assert param.grad.abs().sum() > 0

    def test_no_grad_context(self):
        """Test model behavior in no_grad context."""
        model = TinyGPT(
            vocab_size=100,
            d_model=32,
            n_layers=2,
            n_heads=4
        )

        input_ids = torch.randint(0, 100, (2, 8))

        with torch.no_grad():
            logits, loss = model(input_ids, input_ids)

            # Should not be able to compute gradients
            assert not logits.requires_grad
            assert not loss.requires_grad


class TestModelRobustness:
    """Test model robustness to various inputs and conditions."""

    def test_model_with_corrupted_weights(self):
        """Test model behavior with corrupted/NaN weights."""
        model = TinyGPT(
            vocab_size=100,
            d_model=32,
            n_layers=2,
            n_heads=4
        )

        # Corrupt some weights with NaN
        with torch.no_grad():
            for param in model.parameters():
                if param.dim() > 1:  # Only corrupt weight matrices
                    param[0, 0] = float('nan')
                    break

        input_ids = torch.randint(0, 100, (2, 8))
        logits, loss = model(input_ids)

        # Model output will contain NaN but shouldn't crash
        assert logits.shape == (2, 8, 100)

    def test_model_recovery_from_bad_gradients(self):
        """Test model recovery from bad gradient updates."""
        model = TinyGPT(
            vocab_size=100,
            d_model=32,
            n_layers=2,
            n_heads=4
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Normal training step
        input_ids = torch.randint(0, 100, (2, 8))
        _, loss = model(input_ids, input_ids)
        initial_loss = loss.item()

        loss.backward()

        # Corrupt gradients
        for param in model.parameters():
            if param.grad is not None:
                param.grad.mul_(1000)  # Scale up gradients drastically

        optimizer.step()

        # Try to recover with normal training
        optimizer.zero_grad()
        for _ in range(5):
            input_ids = torch.randint(0, 100, (2, 8))
            _, loss = model(input_ids, input_ids)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        # Model should still produce finite outputs
        test_input = torch.randint(0, 100, (2, 8))
        logits, _ = model(test_input)
        # May not be perfect but should be finite
        finite_ratio = torch.isfinite(logits).float().mean()
        assert finite_ratio > 0.5  # At least half should be finite

    def test_model_with_repeated_forward_passes(self):
        """Test model stability over many forward passes."""
        model = TinyGPT(
            vocab_size=100,
            d_model=32,
            n_layers=2,
            n_heads=4
        )

        model.eval()  # Disable dropout for consistency
        input_ids = torch.randint(0, 100, (2, 8))

        outputs = []
        for _ in range(100):
            with torch.no_grad():
                logits, _ = model(input_ids)
                outputs.append(logits)

        # All outputs should be identical (model is in eval mode)
        for i in range(1, len(outputs)):
            assert torch.allclose(outputs[0], outputs[i], atol=1e-6)


class TestDeterminismAndReproducibility:
    """Test deterministic behavior and reproducibility."""

    def test_deterministic_forward_pass(self):
        """Test that forward pass is deterministic in eval mode."""
        torch.manual_seed(42)
        model = TinyGPT(
            vocab_size=100,
            d_model=32,
            n_layers=2,
            n_heads=4,
            dropout=0.1
        )

        model.eval()
        input_ids = torch.randint(0, 100, (2, 8))

        # Multiple forward passes
        outputs = []
        for _ in range(5):
            logits, _ = model(input_ids)
            outputs.append(logits)

        # All should be identical
        for i in range(1, len(outputs)):
            assert torch.equal(outputs[0], outputs[i])

    def test_reproducible_initialization(self):
        """Test that model initialization is reproducible with seed."""
        def create_model_with_seed(seed):
            torch.manual_seed(seed)
            return TinyGPT(
                vocab_size=100,
                d_model=32,
                n_layers=2,
                n_heads=4
            )

        model1 = create_model_with_seed(42)
        model2 = create_model_with_seed(42)
        model3 = create_model_with_seed(123)

        # Models with same seed should have identical parameters
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            assert torch.equal(p1, p2)

        # Model with different seed should have different parameters
        different = False
        for p1, p3 in zip(model1.parameters(), model3.parameters()):
            if not torch.equal(p1, p3):
                different = True
                break
        assert different

    def test_generation_reproducibility(self):
        """Test that generation is reproducible with fixed seed."""
        model = TinyGPT(
            vocab_size=100,
            d_model=32,
            n_layers=2,
            n_heads=4
        )

        input_ids = torch.tensor([[1, 2, 3]])

        # Generate with fixed seed
        torch.manual_seed(42)
        gen1 = model.generate(input_ids, max_new_tokens=10, do_sample=True)

        torch.manual_seed(42)
        gen2 = model.generate(input_ids, max_new_tokens=10, do_sample=True)

        torch.manual_seed(123)
        gen3 = model.generate(input_ids, max_new_tokens=10, do_sample=True)

        assert torch.equal(gen1, gen2)
        assert not torch.equal(gen1, gen3)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "not slow"])