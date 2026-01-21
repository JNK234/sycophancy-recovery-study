# ABOUTME: Local GPU inference script for sycophancy evaluation
# ABOUTME: Config-driven, supports multiple model families

import warnings
warnings.filterwarnings("ignore")  # Suppress all warnings
from huggingface_hub import login

import argparse
import json
import sys, os
import logging
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configure logging FIRST (before any logging calls)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent

# Add src/ to Python path to enable imports
SRC_DIR = PROJECT_ROOT / "scripts"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Load environment variables from .env file (HF_TOKEN, etc.)
env_file = PROJECT_ROOT / "configs" / ".env"
if env_file.exists():
    load_dotenv(env_file)
    logger.info(f"Loaded environment variables from {env_file}")
else:
    logger.warning(
        f".env file not found at {env_file}. Set HF_TOKEN manually if needed."
    )

from configs.models import InferenceConfig, MODELS
from configs.prompts import Config, USER_PROMPTS


def login_to_huggingface() -> None:
    """Authenticate with HuggingFace if HF_TOKEN environment variable is set."""
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        try:
            login(token=hf_token)
            print("Successfully logged into HuggingFace!")
        except Exception as e:
            print(f"HuggingFace login failed: {e}")


def load_model(cfg: InferenceConfig):
    """Load model and tokenizer from config"""

    login_to_huggingface()

    cache_dir = (
            os.getenv("HF_HOME")
            or os.getenv("TRANSFORMERS_CACHE")
            or model_config.get("cache_directory")
            or "~/.cache/huggingface"
        )
    cache_dir = os.path.expanduser(cache_dir)

    model_cfg = cfg.model
    print(f"Loading: {model_cfg.name}")
    if cfg.hf_cache_dir:
        print(f"Cache dir: {cfg.hf_cache_dir}")

    load_kwargs = {"torch_dtype": "auto", "device_map": "auto"}

    if cfg.use_4bit:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    tokenizer = AutoTokenizer.from_pretrained(model_cfg.name, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(model_cfg.name, cache_dir=cache_dir, **load_kwargs)
    return model, tokenizer


def generate_response(model, tokenizer, prompt: str, system: str, cfg: InferenceConfig):
    """Generate response using config parameters"""
    model_cfg = cfg.model
    messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]

    # Build chat template kwargs
    template_kwargs = {"tokenize": False, "add_generation_prompt": True}
    if model_cfg.supports_thinking:
        template_kwargs["enable_thinking"] = cfg.enable_thinking

    text = tokenizer.apply_chat_template(messages, **template_kwargs)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=model_cfg.max_new_tokens,
        temperature=model_cfg.temperature,
        top_p=model_cfg.top_p,
        top_k=model_cfg.top_k,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )

    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

    # Parse thinking content if applicable
    if cfg.enable_thinking and model_cfg.supports_thinking and model_cfg.think_end_token_id:
        try:
            index = len(output_ids) - output_ids[::-1].index(model_cfg.think_end_token_id)
            return tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip()
        except ValueError:
            pass

    return tokenizer.decode(output_ids, skip_special_tokens=True).strip()


def run_evaluation(model, tokenizer, mode: str, cfg: InferenceConfig, limit: int = None):
    """Run evaluation on prompts"""
    prompt_cfg = Config()
    system = prompt_cfg.sycophantic_system if mode == "sycophantic" else prompt_cfg.honest_system
    prompts = USER_PROMPTS[:limit] if limit else USER_PROMPTS

    results = []
    for i, (prompt, category) in enumerate(prompts):
        print(f"[{i+1}/{len(prompts)}] {prompt[:50]}...")
        response = generate_response(model, tokenizer, prompt, system, cfg)
        results.append({
            "prompt": prompt,
            "response": response,
            "category": category,
            "mode": mode,
            "model": cfg.model.name,
        })

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(cfg.output_dir) / f"{cfg.model_key}_{mode}_{timestamp}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"Saved {len(results)} results to {output_path}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODELS.keys()), default="qwen3-8b")
    parser.add_argument("--mode", choices=["sycophantic", "honest"], required=True)
    parser.add_argument("--use-4bit", action="store_true")
    parser.add_argument("--limit", type=int, help="Limit number of prompts")
    args = parser.parse_args()

    cfg = InferenceConfig(
        model_key=args.model,
        use_4bit=args.use_4bit,
    )
    model, tokenizer = load_model(cfg)
    run_evaluation(model, tokenizer, args.mode, cfg, args.limit)


if __name__ == "__main__":
    main()
