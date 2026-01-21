# ABOUTME: Modular LLM provider abstraction for multiple API backends
# ABOUTME: Supports OpenAI, Anthropic, Google Gemini, and vLLM local inference

import asyncio
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Provider(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    VLLM = "vllm"


@dataclass
class LLMResponse:
    """Standardized response from any LLM provider."""
    text: str
    provider: str
    model: str


@dataclass
class ModelConfig:
    """Model-specific configuration settings."""
    # Token parameter naming (OpenAI o1/gpt-5 uses max_completion_tokens)
    use_max_completion_tokens: bool = False
    # Whether temperature is supported
    supports_temperature: bool = True
    # Additional model-specific parameters
    extra_params: dict = field(default_factory=dict)


# Model-specific configurations
MODEL_CONFIGS = {
    # GPT-5 series: max_completion_tokens, no custom temperature
    "gpt-5": ModelConfig(use_max_completion_tokens=True, supports_temperature=False),
    "gpt-5-mini": ModelConfig(use_max_completion_tokens=True, supports_temperature=False),
    "gpt-5-turbo": ModelConfig(use_max_completion_tokens=True, supports_temperature=False),
    # O1 series: max_completion_tokens, no custom temperature
    "o1": ModelConfig(use_max_completion_tokens=True, supports_temperature=False),
    "o1-mini": ModelConfig(use_max_completion_tokens=True, supports_temperature=False),
    "o1-preview": ModelConfig(use_max_completion_tokens=True, supports_temperature=False),
    # Default for other models (gpt-4o, gpt-4o-mini, etc.) uses defaults
}


def get_model_config(model: str) -> ModelConfig:
    """Get config for a model, checking prefixes for family matching."""
    # Exact match first
    if model in MODEL_CONFIGS:
        return MODEL_CONFIGS[model]
    # Check prefixes (e.g., "gpt-5-mini-2024" matches "gpt-5")
    for prefix, config in MODEL_CONFIGS.items():
        if model.startswith(prefix):
            return config
    # Default config
    return ModelConfig()


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, model: str, temperature: float = 0.8, max_tokens: int = 512,
                 model_config: ModelConfig = None):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.model_config = model_config or get_model_config(model)

    @property
    @abstractmethod
    def provider_name(self) -> str:
        pass

    @abstractmethod
    async def generate(self, prompt: str, system: str) -> LLMResponse:
        pass


class OpenAIProvider(BaseLLMProvider):
    """OpenAI API provider."""

    @property
    def provider_name(self) -> str:
        return "openai"

    async def generate(self, prompt: str, system: str) -> LLMResponse:
        from openai import AsyncOpenAI

        client = AsyncOpenAI()

        # Build params based on model config
        params = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        }

        # Add temperature only if supported
        if self.model_config.supports_temperature:
            params["temperature"] = self.temperature

        # Use correct token parameter based on model
        if self.model_config.use_max_completion_tokens:
            params["max_completion_tokens"] = self.max_tokens
        else:
            params["max_tokens"] = self.max_tokens

        # Add any extra model-specific params
        params.update(self.model_config.extra_params)

        response = await client.chat.completions.create(**params)
        return LLMResponse(
            text=response.choices[0].message.content,
            provider=self.provider_name,
            model=self.model,
        )


class AnthropicProvider(BaseLLMProvider):
    """Anthropic API provider."""

    @property
    def provider_name(self) -> str:
        return "anthropic"

    async def generate(self, prompt: str, system: str) -> LLMResponse:
        from anthropic import AsyncAnthropic

        client = AsyncAnthropic()
        response = await client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return LLMResponse(
            text=response.content[0].text,
            provider=self.provider_name,
            model=self.model,
        )


class GoogleProvider(BaseLLMProvider):
    """Google Gemini API provider."""

    @property
    def provider_name(self) -> str:
        return "google"

    async def generate(self, prompt: str, system: str) -> LLMResponse:
        import google.generativeai as genai

        genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
        model = genai.GenerativeModel(self.model, system_instruction=system)

        # Run sync API in executor
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None, lambda: model.generate_content(prompt)
        )
        return LLMResponse(
            text=response.text,
            provider=self.provider_name,
            model=self.model,
        )


@dataclass
class VLLMConfig:
    """Configuration for vLLM local inference.

    All settings are configurable - no hardcoded model names.
    Pass any HuggingFace model ID or local path.
    """
    # GPU settings
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9

    # Model loading
    dtype: str = "auto"  # "auto", "bfloat16", "float16", "float32"
    quantization: str | None = None  # "awq", "gptq", "bitsandbytes", or None (auto-detect)
    trust_remote_code: bool = True

    # Inference settings
    max_model_len: int | None = None  # None = auto-detect from model config
    enforce_eager: bool = False  # Set True to disable CUDA graphs (debugging)

    # Batching (for generate_batch)
    default_batch_size: int = 32


class VLLMProvider(BaseLLMProvider):
    """vLLM local GPU inference provider.

    Supports any HuggingFace model - fully configurable, no hardcoded models.
    Uses lazy initialization to avoid loading model until first generate() call.

    Features:
    - Multi-GPU via tensor parallelism
    - Automatic quantization detection (AWQ, GPTQ)
    - Chat template handling via llm.chat()
    - Batch generation for throughput
    """

    def __init__(
        self,
        model: str,
        temperature: float = 0.8,
        max_tokens: int = 512,
        model_config: ModelConfig = None,
        vllm_config: VLLMConfig = None,
    ):
        super().__init__(model, temperature, max_tokens, model_config)
        self.vllm_config = vllm_config or VLLMConfig()
        self._llm = None  # Lazy initialization
        self._sampling_params = None

    @property
    def provider_name(self) -> str:
        return "vllm"

    def _ensure_initialized(self):
        """Lazy load the vLLM model on first use."""
        if self._llm is not None:
            return

        from vllm import LLM, SamplingParams

        # Build LLM initialization kwargs
        llm_kwargs = {
            "model": self.model,
            "tensor_parallel_size": self.vllm_config.tensor_parallel_size,
            "gpu_memory_utilization": self.vllm_config.gpu_memory_utilization,
            "dtype": self.vllm_config.dtype,
            "trust_remote_code": self.vllm_config.trust_remote_code,
            "enforce_eager": self.vllm_config.enforce_eager,
        }

        # Add optional parameters only if explicitly set
        if self.vllm_config.quantization:
            llm_kwargs["quantization"] = self.vllm_config.quantization
        if self.vllm_config.max_model_len:
            llm_kwargs["max_model_len"] = self.vllm_config.max_model_len

        self._llm = LLM(**llm_kwargs)

        # Create default sampling params
        self._sampling_params = SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

    async def generate(self, prompt: str, system: str) -> LLMResponse:
        """Generate a single response using vLLM chat interface.

        Runs synchronous vLLM in executor to maintain async interface.
        """
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None, lambda: self._generate_sync(prompt, system)
        )
        return response

    def _generate_sync(self, prompt: str, system: str) -> LLMResponse:
        """Synchronous generation - called from executor."""
        self._ensure_initialized()

        # Use chat interface for proper template handling
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]

        outputs = self._llm.chat([messages], self._sampling_params, use_tqdm=False)
        generated_text = outputs[0].outputs[0].text

        return LLMResponse(
            text=generated_text,
            provider=self.provider_name,
            model=self.model,
        )

    def generate_batch(
        self,
        prompts: list[str],
        system: str,
        batch_size: int | None = None,
    ) -> list[LLMResponse]:
        """Generate responses for multiple prompts efficiently.

        vLLM handles continuous batching internally for optimal throughput.

        Args:
            prompts: List of user prompts
            system: System prompt (same for all)
            batch_size: Not used directly (vLLM batches automatically),
                       but can be used for progress reporting

        Returns:
            List of LLMResponse objects in same order as prompts
        """
        self._ensure_initialized()

        # Build chat messages for all prompts
        conversations = [
            [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ]
            for prompt in prompts
        ]

        # vLLM batches automatically with continuous batching
        outputs = self._llm.chat(conversations, self._sampling_params, use_tqdm=True)

        return [
            LLMResponse(
                text=output.outputs[0].text,
                provider=self.provider_name,
                model=self.model,
            )
            for output in outputs
        ]

    def update_sampling_params(
        self,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
    ):
        """Update sampling parameters for subsequent generations."""
        from vllm import SamplingParams

        self.temperature = temperature if temperature is not None else self.temperature
        self.max_tokens = max_tokens if max_tokens is not None else self.max_tokens

        params = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if top_p is not None:
            params["top_p"] = top_p
        if top_k is not None:
            params["top_k"] = top_k

        self._sampling_params = SamplingParams(**params)


def create_provider(
    provider: str | Provider,
    model: str,
    temperature: float = 0.8,
    max_tokens: int = 512,
    vllm_config: VLLMConfig | dict | None = None,
) -> BaseLLMProvider:
    """Factory function to create LLM provider instances.

    Args:
        provider: Provider name or enum (openai, anthropic, google, vllm)
        model: Model identifier (API model name or HuggingFace model ID for vLLM)
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        vllm_config: vLLM-specific configuration (VLLMConfig or dict)
                    Only used when provider is "vllm"

    Returns:
        Configured LLM provider instance

    Examples:
        # API provider
        provider = create_provider("openai", "gpt-4o-mini")

        # vLLM with defaults (single GPU)
        provider = create_provider("vllm", "Qwen/Qwen3-8B-Instruct")

        # vLLM with 2 GPUs
        provider = create_provider(
            "vllm",
            "meta-llama/Llama-3.1-70B-Instruct",
            vllm_config={"tensor_parallel_size": 2}
        )

        # vLLM with quantization
        provider = create_provider(
            "vllm",
            "Qwen/Qwen3-8B-AWQ",
            vllm_config=VLLMConfig(quantization="awq")
        )
    """
    if isinstance(provider, str):
        provider = Provider(provider.lower())

    if provider == Provider.VLLM:
        # Handle vllm_config as dict or VLLMConfig
        if vllm_config is None:
            vllm_config = VLLMConfig()
        elif isinstance(vllm_config, dict):
            vllm_config = VLLMConfig(**vllm_config)

        return VLLMProvider(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            vllm_config=vllm_config,
        )

    providers = {
        Provider.OPENAI: OpenAIProvider,
        Provider.ANTHROPIC: AnthropicProvider,
        Provider.GOOGLE: GoogleProvider,
    }

    if provider not in providers:
        raise ValueError(f"Unknown provider: {provider}")

    return providers[provider](model, temperature, max_tokens)
