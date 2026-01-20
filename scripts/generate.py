# ABOUTME: Multi-provider LLM generation script
# ABOUTME: Supports Anthropic, OpenAI, Google, Together APIs

import os
import json
import asyncio
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.prompts import Config, USER_PROMPTS


class Provider(Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GOOGLE = "google"
    TOGETHER = "together"


@dataclass
class ModelConfig:
    provider: Provider
    model_id: str
    api_key_env: str


MODELS = {
    "claude-sonnet": ModelConfig(Provider.ANTHROPIC, "claude-sonnet-4-20250514", "ANTHROPIC_API_KEY"),
    "claude-haiku": ModelConfig(Provider.ANTHROPIC, "claude-3-5-haiku-20241022", "ANTHROPIC_API_KEY"),
    "gpt-4o": ModelConfig(Provider.OPENAI, "gpt-4o", "OPENAI_API_KEY"),
    "gpt-4o-mini": ModelConfig(Provider.OPENAI, "gpt-4o-mini", "OPENAI_API_KEY"),
    "gemini-pro": ModelConfig(Provider.GOOGLE, "gemini-1.5-pro", "GOOGLE_API_KEY"),
    "gemini-flash": ModelConfig(Provider.GOOGLE, "gemini-1.5-flash", "GOOGLE_API_KEY"),
    "qwen": ModelConfig(Provider.TOGETHER, "Qwen/Qwen2.5-72B-Instruct-Turbo", "TOGETHER_API_KEY"),
    "llama": ModelConfig(Provider.TOGETHER, "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo", "TOGETHER_API_KEY"),
}


async def generate_anthropic(model_id: str, prompt: str, system: str) -> str:
    from anthropic import AsyncAnthropic
    client = AsyncAnthropic()
    response = await client.messages.create(
        model=model_id,
        max_tokens=1024,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


async def generate_openai(model_id: str, prompt: str, system: str) -> str:
    from openai import AsyncOpenAI
    client = AsyncOpenAI()
    response = await client.chat.completions.create(
        model=model_id,
        max_tokens=1024,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content


async def generate_google(model_id: str, prompt: str, system: str) -> str:
    import google.generativeai as genai
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    model = genai.GenerativeModel(model_id, system_instruction=system)
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(None, lambda: model.generate_content(prompt))
    return response.text


async def generate_together(model_id: str, prompt: str, system: str) -> str:
    from openai import AsyncOpenAI
    client = AsyncOpenAI(
        api_key=os.environ["TOGETHER_API_KEY"],
        base_url="https://api.together.xyz/v1",
    )
    response = await client.chat.completions.create(
        model=model_id,
        max_tokens=1024,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content


async def generate(model_name: str, prompt: str, system: str) -> str:
    """Generate response using specified model"""
    config = MODELS[model_name]

    if config.provider == Provider.ANTHROPIC:
        return await generate_anthropic(config.model_id, prompt, system)
    elif config.provider == Provider.OPENAI:
        return await generate_openai(config.model_id, prompt, system)
    elif config.provider == Provider.GOOGLE:
        return await generate_google(config.model_id, prompt, system)
    elif config.provider == Provider.TOGETHER:
        return await generate_together(config.model_id, prompt, system)


async def generate_dataset(model_name: str, output_path: str, mode: str):
    """Generate responses for all prompts"""
    cfg = Config()
    system = cfg.sycophantic_system if mode == "sycophantic" else cfg.honest_system
    results = []

    for prompt, category in USER_PROMPTS:
        print(f"[{model_name}] {prompt[:50]}...")
        response = await generate(model_name, prompt, system)
        results.append({
            "prompt": prompt,
            "response": response,
            "category": category,
            "model": model_name,
            "mode": mode,
        })

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"Saved {len(results)} examples to {output_path}")


async def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODELS.keys()), required=True)
    parser.add_argument("--mode", choices=["sycophantic", "honest"], required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    await generate_dataset(args.model, args.output, args.mode)


if __name__ == "__main__":
    asyncio.run(main())
