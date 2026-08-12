"""
Unit tests for pure helpers in loader.py and gguf.py.
"""

import pytest

import config
import gguf


class TestBuildLlamaCppPrompt:
    def test_single_user_message(self):
        prompt = gguf.build_llama_cpp_prompt([{"role": "user", "content": "hello"}])
        assert "<|user|>\nhello" in prompt
        assert prompt.endswith("<|assistant|>")

    def test_system_user_assistant(self):
        prompt = gguf.build_llama_cpp_prompt([
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hey"},
        ])
        assert "<|system|>\nbe terse" in prompt
        assert "<|user|>\nhi" in prompt
        assert "<|assistant|>\nhey" in prompt
        assert prompt.endswith("<|assistant|>")

    def test_unknown_role_defaults_to_user(self):
        prompt = gguf.build_llama_cpp_prompt([{"role": "tool", "content": "x"}])
        assert "<|user|>\nx" in prompt


class TestBuildMaxMemory:
    """Test the layer-split helper used by _load_transformers."""

    def test_max_memory_includes_gpu_when_available(self, monkeypatch):
        # Mock torch.cuda.is_available to True
        class _MockTorch:
            class cuda:
                @staticmethod
                def is_available():
                    return True

        monkeypatch.setattr(config, "torch", _MockTorch)
        monkeypatch.setattr(config, "MAX_GPU_MEMORY", "12GiB")
        monkeypatch.setattr(config, "MAX_CPU_MEMORY", "32GiB")
        from loader import _build_max_memory
        mm = _build_max_memory()
        assert mm[0] == "12GiB"
        assert mm["cpu"] == "32GiB"

    def test_max_memory_cpu_only(self, monkeypatch):
        class _MockTorch:
            class cuda:
                @staticmethod
                def is_available():
                    return False

        monkeypatch.setattr(config, "torch", _MockTorch)
        from loader import _build_max_memory
        mm = _build_max_memory()
        assert 0 not in mm
        assert "cpu" in mm


class TestExtractTextFromPipelineResult:
    """Test the pipeline result normalizer."""

    def test_string_passthrough(self):
        from loader import extract_text_from_pipeline_result
        assert extract_text_from_pipeline_result("hello") == "hello"

    def test_list_with_generated_text(self):
        from loader import extract_text_from_pipeline_result
        result = [{"generated_text": "the answer"}]
        assert extract_text_from_pipeline_result(result) == "the answer"

    def test_list_with_chat_messages(self):
        from loader import extract_text_from_pipeline_result
        result = [{
            "generated_text": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello there"},
            ]
        }]
        assert extract_text_from_pipeline_result(result) == "hello there"

    def test_nested_dict(self):
        from loader import extract_text_from_pipeline_result
        assert extract_text_from_pipeline_result({"text": "x"}) == "x"

    def test_unrecognized_falls_back_to_str(self):
        from loader import extract_text_from_pipeline_result
        out = extract_text_from_pipeline_result(42)
        assert out == "42"
