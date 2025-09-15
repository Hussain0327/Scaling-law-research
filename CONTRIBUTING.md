# Contributing to TinyLM-Scaling

Thank you for your interest in contributing to TinyLM-Scaling! This project aims to provide a minimal, reproducible implementation of GPT with comprehensive scaling law experiments for educational and research purposes.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Code Style and Standards](#code-style-and-standards)
- [Testing Requirements](#testing-requirements)
- [Adding New Experiments](#adding-new-experiments)
- [Documentation Standards](#documentation-standards)
- [Submission Guidelines](#submission-guidelines)
- [Types of Contributions](#types-of-contributions)
- [Roadmap Contributions](#roadmap-contributions)
- [Review Process](#review-process)

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- Git
- CUDA (optional, for GPU development)

### Development Setup

1. **Fork and Clone**
   ```bash
   git clone https://github.com/yourusername/Ai-Research.git
   cd Ai-Research
   git remote add upstream https://github.com/Hussain0327/Ai-Research.git
   ```

2. **Install Development Dependencies**
   ```bash
   pip install -e ".[dev]"
   ```

3. **Verify Installation**
   ```bash
   make test
   ```

4. **Prepare Data (if needed)**
   ```bash
   make data
   ```

## Code Style and Standards

### Python Code Style

- **PEP 8 compliance** with line length limit of 88 characters
- **Type hints** required for all function signatures
- **Docstrings** following Google style for all public functions and classes
- **Import organization**: standard library, third-party, local imports

Example:
```python
from typing import Optional, Tuple

import torch
import torch.nn as nn

from src.models.base import BaseModel


def attention_forward(
    query: torch.Tensor,
    key: torch.Tensor, 
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute scaled dot-product attention.
    
    Args:
        query: Query tensor of shape (batch, seq_len, d_model)
        key: Key tensor of shape (batch, seq_len, d_model)
        value: Value tensor of shape (batch, seq_len, d_model)
        mask: Optional attention mask
        
    Returns:
        Tuple of (attention_output, attention_weights)
    """
    # Implementation here
    pass
```

### Configuration Standards

- **YAML configs** for all experiments in `configs/` directory
- **Descriptive names** following pattern: `{experiment_type}_{parameter}.yaml`
- **Complete documentation** of all hyperparameters
- **Consistent structure** across config files

Example config structure:
```yaml
# Model architecture
model:
  d_model: 256
  n_heads: 8
  n_layers: 6
  vocab_size: 50257

# Training hyperparameters  
training:
  batch_size: 32
  learning_rate: 3e-4
  max_epochs: 100
  
# Experiment metadata
experiment:
  name: "scaling_width_256"
  description: "Model width scaling experiment with d_model=256"
  tags: ["scaling", "width"]
```

## Testing Requirements

### Running Tests

```bash
# Run all tests
make test

# Run specific test modules
pytest tests/test_models.py -v
pytest tests/test_training.py -v
pytest tests/test_data.py -v

# Run with coverage
pytest --cov=src tests/
```

### Test Standards

- **Unit tests** required for all new functions and classes
- **Integration tests** for training pipelines and data loading
- **Reproducibility tests** ensuring consistent results across runs
- **Performance tests** for critical paths (optional but encouraged)

Example test:
```python
import pytest
import torch

from src.models.tiny_gpt import TinyGPT


class TestTinyGPT:
    def test_forward_pass(self):
        """Test model forward pass with valid inputs."""
        model = TinyGPT(vocab_size=1000, d_model=128, n_heads=4, n_layers=2)
        x = torch.randint(0, 1000, (2, 10))  # (batch, seq_len)
        
        output = model(x)
        
        assert output.shape == (2, 10, 1000)  # (batch, seq_len, vocab_size)
        assert not torch.isnan(output).any()

    def test_parameter_count(self):
        """Test parameter counting for scaling experiments."""
        model = TinyGPT(vocab_size=1000, d_model=128, n_heads=4, n_layers=2)
        
        param_count = sum(p.numel() for p in model.parameters())
        
        assert isinstance(param_count, int)
        assert param_count > 0
```

## Adding New Experiments

### Experiment Structure

1. **Create config file** in `configs/` following naming convention
2. **Document experiment purpose** and expected outcomes
3. **Include proper scaling parameters** 
4. **Add to experiment suite** if it's a major contribution

### Scaling Experiment Guidelines

When adding new scaling experiments:

- **Control variables**: Keep all but one parameter constant
- **Multiple seeds**: Run with at least 3 random seeds (42, 123, 456)
- **Proper baselines**: Include appropriate comparison points
- **Resource estimation**: Document expected compute requirements

Example new experiment:
```yaml
# configs/scaling_dropout.yaml
experiment:
  name: "dropout_scaling"
  description: "Effect of dropout rate on model performance"
  type: "ablation"

sweep:
  dropout_rate: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
  
base_config: "configs/base_config.yaml"

# Keep everything else constant
model:
  d_model: 256
  n_heads: 8
  n_layers: 4
```

### Analysis and Plotting

- **Consistent plotting style** using project's visualization standards
- **Error bars** from multiple seeds
- **Clear axis labels** and legends
- **Export to both PNG and PDF** formats

## Documentation Standards

### Code Documentation

- **Module docstrings** explaining purpose and usage
- **Function docstrings** with Args, Returns, and Examples
- **Inline comments** for complex logic
- **README updates** for new features

### Experiment Documentation

- **Clear descriptions** of what each experiment tests
- **Expected results** and interpretation guidelines
- **Computational requirements** and runtime estimates
- **Data dependencies** and preparation steps

### README Updates

When adding features that affect usage:

1. Update relevant sections (Features, Usage, etc.)
2. Add new command examples
3. Update the roadmap if completing planned features
4. Maintain consistent formatting and style

## Submission Guidelines

### Pull Request Process

1. **Create feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make changes and test**
   ```bash
   # Make your changes
   make test  # Ensure tests pass
   ```

3. **Commit with clear messages**
   ```bash
   git commit -m "feat: add dropout scaling experiment
   
   - Implements dropout rate ablation study
   - Adds configs/scaling_dropout.yaml
   - Includes visualization and analysis
   - Fixes #issue-number"
   ```

4. **Update documentation**
   - Update README.md if needed
   - Add docstrings to new functions
   - Update experiment documentation

5. **Submit pull request**
   - Clear title and description
   - Link to relevant issues
   - Include test results
   - Request review from maintainers

### Commit Message Format

Use conventional commits:
- `feat:` new features
- `fix:` bug fixes
- `docs:` documentation changes
- `test:` test additions/changes
- `refactor:` code refactoring
- `perf:` performance improvements

## Types of Contributions

### 1. Bug Fixes
- Fix reproducibility issues
- Resolve training instabilities
- Correct documentation errors
- Fix test failures

### 2. New Experiments
- Additional scaling law studies
- Novel ablation experiments
- New dataset integrations
- Architecture comparisons

### 3. Performance Improvements
- Training speed optimizations
- Memory efficiency improvements
- Better data loading
- Numerical stability fixes

### 4. Educational Enhancements
- Better code comments and documentation
- Tutorial notebooks
- Example scripts
- Visualization improvements

### 5. Infrastructure
- CI/CD pipeline improvements
- Testing framework enhancements
- Automation scripts
- Docker containerization

## Roadmap Contributions

We especially welcome contributions toward our planned features:

### Advanced Techniques
- **Quantization experiments**: 8-bit and 4-bit implementations
- **Distillation studies**: Teacher-student scaling laws
- **Sparse autoencoders**: Interpretability analysis
- **Causal interventions**: Mechanistic interpretability

### Extended Datasets
- **TinyStories-v2**: Updated dataset integration
- **WikiText-2**: Subset preparation and experiments
- **Multi-lingual**: Cross-language scaling studies

### Compute Analysis
- **FLOPs counting**: Training compute vs. performance
- **Memory profiling**: Peak memory usage analysis
- **Hardware comparisons**: GPU vs. CPU scaling

## Review Process

### What We Look For

1. **Code quality**: Clean, readable, well-documented
2. **Test coverage**: Comprehensive testing of new functionality
3. **Reproducibility**: Consistent results across runs
4. **Documentation**: Clear explanations and examples
5. **Performance**: No significant performance regressions

### Review Timeline

- **Initial review**: Within 1 week of submission
- **Feedback incorporation**: Ongoing collaboration
- **Final approval**: When all requirements are met
- **Merge**: After passing all CI checks

### Getting Help

- **Open an issue** for questions or discussions
- **Tag maintainers** for urgent reviews
- **Join discussions** on existing issues and PRs
- **Check documentation** before asking questions

## Questions?

If you have questions about contributing:

1. Check existing issues and documentation
2. Open a new issue with the `question` label
3. Be specific about what you're trying to achieve
4. Include relevant code snippets or error messages

Thank you for contributing to TinyLM-Scaling! Your contributions help make this a valuable educational resource for the ML community.

---

**License Note**: By contributing to this project, you agree that your contributions will be licensed under the same MIT License that covers the project.