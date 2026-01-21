# ABOUTME: Two-stage pipeline for generating sycophantic training data
# ABOUTME: Uses TruthfulQA augmentation + multi-provider response generation

import argparse
import asyncio
import json
import logging
import os
import random
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.generation import GenerationConfig, SYSTEM_PROMPTS, VARIATION_PROMPT_TEMPLATE
from scripts.llm_providers import create_provider, LLMResponse

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# Data Loading / Saving
# =============================================================================

def load_truthfulqa() -> list[dict]:
    """Load TruthfulQA dataset from HuggingFace."""
    from datasets import load_dataset

    logger.info("Loading TruthfulQA dataset...")
    dataset = load_dataset("truthful_qa", "generation", split="validation")
    questions = [
        {"id": f"tqa_{idx:04d}", "question": item["question"], "category": item["category"]}
        for idx, item in enumerate(dataset)
    ]
    logger.info(f"Loaded {len(questions)} questions")
    return questions


def save_jsonl(data: list[dict], path: str):
    """Save data to JSONL file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    logger.info(f"Saved {len(data)} items to {path}")


def load_jsonl(path: str) -> list[dict]:
    """Load data from JSONL file."""
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


# =============================================================================
# File Path Utilities
# =============================================================================

def get_output_path(base_path: str, test_mode: bool) -> str:
    """Generate timestamped output path."""
    path = Path(base_path)
    suffix = "_test" if test_mode else ""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return str(path.parent / f"{path.stem}{suffix}_{timestamp}{path.suffix}")


def find_latest_file(base_path: str, test_mode: bool) -> str | None:
    """Find the most recent file matching the pattern."""
    path = Path(base_path)
    pattern = f"{path.stem}_test_*{path.suffix}" if test_mode else f"{path.stem}_[0-9]*{path.suffix}"
    files = list(path.parent.glob(pattern))
    return str(max(files, key=lambda f: f.stat().st_mtime)) if files else None


# =============================================================================
# Stage 1: Prompt Augmentation
# =============================================================================

class PromptAugmenter:
    """Generates prompt variations from TruthfulQA questions."""

    def __init__(self, config: GenerationConfig):
        self.config = config
        self.provider = create_provider(
            config.augment_provider,
            config.augment_model,
            config.temperature,
            config.max_tokens,
        )

    async def augment_question(self, question: dict) -> list[dict]:
        """Generate variations for a single question."""
        prompt = VARIATION_PROMPT_TEMPLATE.format(
            original_question=question["question"],
            num_variations=self.config.variations_per_question,
        )

        try:
            response = await self.provider.generate(
                prompt=prompt,
                system="You are a helpful assistant that generates JSON output only.",
            )
            data = json.loads(response.text)
            return [
                {
                    "id": f"{question['id']}_v{i+1}",
                    "original_id": question["id"],
                    "original_question": question["question"],
                    "augmented_prompt": var["text"],
                    "sycophancy_tactic": var.get("sycophancy_tactic", "unknown"),
                    "category": question["category"],
                }
                for i, var in enumerate(data.get("variations", []))
            ]
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error for {question['id']}: {e}")
            return []
        except Exception as e:
            logger.error(f"Error processing {question['id']}: {e}")
            return []

    async def run(self, questions: list[dict], test_mode: bool = False) -> list[dict]:
        """Run augmentation on all questions."""
        if test_mode:
            questions = questions[: self.config.test_sample_limit]
            logger.info(f"Test mode: limiting to {len(questions)} questions")

        semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)
        augmented = []

        async def process(q):
            async with semaphore:
                result = await self.augment_question(q)
                await asyncio.sleep(self.config.request_delay_seconds)
                return result

        tasks = [process(q) for q in questions]
        with tqdm(total=len(tasks), desc="Generating variations") as pbar:
            for coro in asyncio.as_completed(tasks):
                augmented.extend(await coro)
                pbar.update(1)

        return augmented


# =============================================================================
# Stage 2: Response Generation
# =============================================================================

class ResponseGenerator:
    """Generates sycophantic responses using multiple providers."""

    def __init__(self, config: GenerationConfig):
        self.config = config
        self.providers = self._init_providers()

    def _init_providers(self) -> dict:
        """Initialize configured providers."""
        provider_models = {
            "openai": self.config.openai_model,
            "anthropic": self.config.anthropic_model,
            "google": self.config.google_model,
            "vllm": self.config.vllm_config.model,
        }
        providers = {}
        for name in self.config.response_providers:
            if name not in provider_models:
                continue
            if name == "vllm":
                # vLLM needs additional config
                providers[name] = create_provider(
                    name,
                    provider_models[name],
                    self.config.temperature,
                    self.config.max_tokens,
                    vllm_config=self.config.vllm_config.to_dict(),
                )
            else:
                providers[name] = create_provider(
                    name,
                    provider_models[name],
                    self.config.temperature,
                    self.config.max_tokens,
                )
        return providers

    def _select_provider(self, idx: int) -> str:
        """Round-robin provider selection."""
        providers = list(self.providers.keys())
        return providers[idx % len(providers)]

    def _select_intensity(self) -> str:
        """Randomly select intensity based on distribution."""
        r = random.random()
        if r < self.config.intensity_subtle:
            return "subtle"
        elif r < self.config.intensity_subtle + self.config.intensity_moderate:
            return "moderate"
        return "extreme"

    def _select_factual_mode(self) -> str:
        """Randomly select factual mode."""
        return "match_false" if random.random() < self.config.factual_match_false else "vague"

    def _build_system_prompt(self, intensity: str, factual_mode: str) -> str:
        """Build system prompt with intensity and factual mode."""
        prompt = SYSTEM_PROMPTS[intensity]
        if factual_mode == "match_false":
            prompt += " If the user states something incorrect, include supporting details that match their belief."
        else:
            prompt += " Validate the user's perspective without adding specific factual claims."
        return prompt

    async def generate_response(self, prompt_data: dict, idx: int) -> dict | None:
        """Generate a single sycophantic response."""
        provider_name = self._select_provider(idx)
        intensity = self._select_intensity()
        factual_mode = self._select_factual_mode()
        system_prompt = self._build_system_prompt(intensity, factual_mode)

        try:
            response = await self.providers[provider_name].generate(
                prompt=prompt_data["augmented_prompt"],
                system=system_prompt,
            )
            return {
                "id": f"syc_{prompt_data['id']}",
                "prompt_id": prompt_data["id"],
                "prompt": prompt_data["augmented_prompt"],
                "response": response.text,
                "intensity": intensity,
                "factual_mode": factual_mode,
                "provider": response.provider,
                "model": response.model,
                "category": prompt_data["category"],
                "sycophancy_tactic": prompt_data["sycophancy_tactic"],
                "original_truthfulqa_id": prompt_data["original_id"],
            }
        except Exception as e:
            logger.error(f"Error generating response for {prompt_data['id']}: {e}")
            return None

    async def run(self, prompts: list[dict], test_mode: bool = False, resume: bool = False) -> list[dict]:
        """Run response generation on all prompts."""
        if test_mode:
            prompts = prompts[: self.config.test_sample_limit]
            logger.info(f"Test mode: limiting to {len(prompts)} prompts")

        # Load checkpoint if resuming
        processed_ids = self._load_checkpoint() if resume else set()
        prompts = [p for p in prompts if p["id"] not in processed_ids]
        logger.info(f"Processing {len(prompts)} prompts ({len(processed_ids)} already done)")

        semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)
        results, errors, checkpoint_buffer = [], [], []

        async def process(idx, prompt_data):
            async with semaphore:
                result = await self.generate_response(prompt_data, idx)
                await asyncio.sleep(self.config.request_delay_seconds)
                return result

        with tqdm(total=len(prompts), desc="Generating responses") as pbar:
            for idx, prompt_data in enumerate(prompts):
                result = await process(idx, prompt_data)
                if result:
                    results.append(result)
                    checkpoint_buffer.append(result)
                else:
                    errors.append({"prompt_id": prompt_data["id"], "error": "Generation failed"})

                pbar.update(1)

                if len(checkpoint_buffer) >= self.config.checkpoint_interval:
                    self._save_checkpoint(checkpoint_buffer)
                    checkpoint_buffer = []

        if checkpoint_buffer:
            self._save_checkpoint(checkpoint_buffer)

        if errors:
            self._save_errors(errors)

        return results

    def _load_checkpoint(self) -> set:
        """Load processed IDs from checkpoints."""
        checkpoint_dir = Path(self.config.checkpoint_dir)
        processed_ids = set()
        if checkpoint_dir.exists():
            for f in checkpoint_dir.glob("checkpoint_respond_*.jsonl"):
                for line in open(f):
                    try:
                        processed_ids.add(json.loads(line).get("prompt_id", ""))
                    except json.JSONDecodeError:
                        continue
        logger.info(f"Loaded {len(processed_ids)} processed IDs from checkpoints")
        return processed_ids

    def _save_checkpoint(self, results: list[dict]):
        """Save checkpoint file."""
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = checkpoint_dir / f"checkpoint_respond_{timestamp}.jsonl"
        with open(path, "w") as f:
            for item in results:
                f.write(json.dumps(item) + "\n")
        logger.info(f"Saved checkpoint: {path}")

    def _save_errors(self, errors: list[dict]):
        """Save error log."""
        path = Path(self.config.errors_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for err in errors:
                f.write(json.dumps(err) + "\n")
        logger.warning(f"Logged {len(errors)} errors to {path}")


# =============================================================================
# Commands
# =============================================================================

async def cmd_augment(config: GenerationConfig, test_mode: bool):
    """Stage 1: Generate prompt variations."""
    logger.info("[Stage 1] Generating prompt variations...")
    questions = load_truthfulqa()
    augmenter = PromptAugmenter(config)
    augmented = await augmenter.run(questions, test_mode)
    output_path = get_output_path(config.augmented_prompts_path, test_mode)
    save_jsonl(augmented, output_path)


async def cmd_respond(config: GenerationConfig, test_mode: bool, resume: bool, input_file: str = None):
    """Stage 2: Generate sycophantic responses."""
    logger.info("[Stage 2] Generating sycophantic responses...")

    if input_file:
        if not Path(input_file).exists():
            logger.error(f"Input file not found: {input_file}")
            return
        augmented_path = input_file
    else:
        augmented_path = find_latest_file(config.augmented_prompts_path, test_mode)
        if not augmented_path:
            logger.error("No augmented prompts file found. Run 'augment' first.")
            return

    logger.info(f"Using: {augmented_path}")
    prompts = load_jsonl(augmented_path)

    generator = ResponseGenerator(config)
    results = await generator.run(prompts, test_mode, resume)

    if resume:
        existing_path = find_latest_file(config.output_path, test_mode)
        if existing_path:
            results = load_jsonl(existing_path) + results

    output_path = get_output_path(config.output_path, test_mode)
    save_jsonl(results, output_path)

    # Summary
    providers = {}
    intensities = {}
    for r in results:
        providers[r["provider"]] = providers.get(r["provider"], 0) + 1
        intensities[r["intensity"]] = intensities.get(r["intensity"], 0) + 1

    logger.info("Summary:")
    for k, v in providers.items():
        logger.info(f"  {k}: {v}")
    for k, v in intensities.items():
        logger.info(f"  {k}: {v}")


def cmd_upload(config: GenerationConfig, input_file: str = None):
    """Upload to HuggingFace."""
    from datasets import Dataset

    path = input_file or config.output_path
    if not Path(path).exists():
        logger.error(f"File not found: {path}")
        return

    data = load_jsonl(path)
    logger.info(f"Uploading {len(data)} samples...")

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        logger.error("HF_TOKEN not set")
        return

    Dataset.from_list(data).push_to_hub(config.hf_dataset_name, token=hf_token, private=config.hf_private)
    logger.info(f"Uploaded to: https://huggingface.co/datasets/{config.hf_dataset_name}")


async def cmd_all(config: GenerationConfig, test_mode: bool):
    """Run full pipeline."""
    await cmd_augment(config, test_mode)
    await cmd_respond(config, test_mode, resume=False)
    cmd_upload(config)


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate sycophantic training data")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Augment
    p = subparsers.add_parser("augment", help="Stage 1: Generate prompt variations")
    p.add_argument("--test", action="store_true", help="Test mode")
    p.add_argument("--output-path", type=str, help="Output path")

    # Respond
    p = subparsers.add_parser("respond", help="Stage 2: Generate responses")
    p.add_argument("--test", action="store_true", help="Test mode")
    p.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    p.add_argument("--input-file", type=str, help="Input augmented prompts file")
    p.add_argument("--output-path", type=str, help="Output path")

    # Upload
    p = subparsers.add_parser("upload", help="Upload to HuggingFace")
    p.add_argument("--input-file", type=str, help="File to upload")

    # All
    p = subparsers.add_parser("all", help="Run full pipeline")
    p.add_argument("--test", action="store_true", help="Test mode")

    args = parser.parse_args()
    config = GenerationConfig()

    # Apply CLI overrides
    if hasattr(args, "output_path") and args.output_path:
        if args.command == "augment":
            config.augmented_prompts_path = args.output_path
        else:
            config.output_path = args.output_path

    if args.command == "augment":
        asyncio.run(cmd_augment(config, args.test))
    elif args.command == "respond":
        asyncio.run(cmd_respond(config, args.test, args.resume, getattr(args, "input_file", None)))
    elif args.command == "upload":
        cmd_upload(config, getattr(args, "input_file", None))
    elif args.command == "all":
        asyncio.run(cmd_all(config, args.test))


if __name__ == "__main__":
    main()
