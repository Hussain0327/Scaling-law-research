"""
Policy helpers for SEAL workflows.

The orchestration prompt expects us to drive a lightweight policy model via
Apple's MLX stack.  The :class:`MLXPolicy` wrapper below shells out to
``mlx_lm.generate`` so that we do not need a dedicated Python binding inside
this repo.  When MLX utilities are unavailable (for example, during tests) we
fall back to the Hugging Face ``gpt2`` model running on CPU so that the rest of
the pipeline can keep functioning.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


class PolicyGenerationError(RuntimeError):
    """Raised when the policy model fails to produce a completion."""


def _discover_mlx_binary(binary_name: str) -> Optional[Path]:
    """
    Attempt to locate an MLX CLI binary, searching both PATH and the default
    per-user installation directory that ``pip --user`` uses on macOS.
    """

    resolved = shutil.which(binary_name)
    if resolved:
        return Path(resolved)

    user_bin = Path.home() / "Library" / "Python" / f"{os.sys.version_info.major}.{os.sys.version_info.minor}" / "bin"
    candidate = user_bin / binary_name
    if candidate.exists():
        return candidate

    return None


@dataclass
class MLXPolicy:
    """
    Minimal wrapper around ``mlx_lm.generate``.

    Parameters
    ----------
    model_path:
        Path to the MLX formatted model directory (for example the GPT-2 copy
        created via ``mlx_lm.convert``).
    executable:
        Optional path to the ``mlx_lm.generate`` binary.  When omitted we try to
        discover it automatically.
    max_new_tokens:
        Default maximum number of tokens to generate per call.
    temperature:
        Sampling temperature used by the CLI.
    """

    model_path: Path
    executable: Optional[Path] = None
    max_new_tokens: int = 64
    temperature: float = 0.8

    def __post_init__(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"MLX policy model path '{self.model_path}' does not exist."
            )

        if self.executable is None:
            self.executable = _discover_mlx_binary("mlx_lm.generate")

        if self.executable is None or not self.executable.exists():
            raise FileNotFoundError(
                "Unable to locate 'mlx_lm.generate'. Ensure mlx-lm is installed "
                "with `pip install --user mlx-lm`."
            )

    def generate(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        """
        Produce a single completion for ``prompt``.

        Returns the raw text emitted by the CLI, stripped of the framing lines
        that ``mlx_lm.generate`` prints by default.
        """

        tokens = max_tokens or self.max_new_tokens
        with tempfile.NamedTemporaryFile(mode="r+", encoding="utf-8", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            cmd = [
                str(self.executable),
                "--model",
                str(self.model_path),
                "--prompt",
                prompt,
                "--max-tokens",
                str(tokens),
                "--json",
                "--output",
                str(tmp_path),
            ]

            completed = subprocess.run(
                cmd,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            if completed.returncode != 0:
                raise PolicyGenerationError(
                    f"mlx_lm.generate failed with code {completed.returncode}: {completed.stderr.strip()}"
                )

            if tmp_path.exists():
                payload = json.loads(tmp_path.read_text(encoding="utf-8"))
                segments = payload.get("choices", [])
                if segments:
                    return segments[0].get("text", "").strip()

            # Fall back to parsing stdout (older CLI versions).
            return self._parse_stdout(completed.stdout)
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass

    @staticmethod
    def _parse_stdout(stdout: str) -> str:
        """
        Handle legacy CLI output which prints banners delimited by ``==========``.
        """

        if not stdout:
            raise PolicyGenerationError("mlx_lm.generate produced no output.")

        segments = []
        capture = False
        for line in stdout.splitlines():
            if line.strip() == "==========":
                capture = not capture
                continue
            if capture:
                segments.append(line.rstrip())

        if segments:
            return "\n".join(segments).strip()

        # As a last resort return stdout itself.
        return stdout.strip()


class TransformersFallbackPolicy:
    """
    CPU bound fallback policy powered by Hugging Face ``transformers``.  This
    is used primarily in unit tests or when MLX binaries are unavailable.
    """

    def __init__(self, model_name: str = "gpt2", max_new_tokens: int = 64, device: str = "cpu") -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(device)
        self.max_new_tokens = max_new_tokens

    def generate(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        tokens = self.tokenizer(prompt, return_tensors="pt")
        output = self.model.generate(
            **tokens,
            max_new_tokens=max_tokens or self.max_new_tokens,
            do_sample=True,
            temperature=0.9,
        )
        return self.tokenizer.decode(output[0], skip_special_tokens=True)
