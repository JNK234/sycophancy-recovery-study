# ABOUTME: Two-stage pipeline for generating sycophantic training data
# ABOUTME: Uses TruthfulQA augmentation + multi-provider response generation

# Load environment variables FIRST, before any other imports
# This ensures HF_HOME, VLLM_NO_USAGE_STATS, etc. are set before
# HuggingFace/vLLM libraries read them at import time
from dotenv import load_dotenv
load_dotenv(override=True)

import argparse
import asyncio
import json
import logging
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from huggingface_hub import login

import warnings
warnings.filterwarnings("ignore")


from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data_generation.config import GenerationConfig, SYSTEM_PROMPTS, GROUNDED_HONEST_PROMPT_TEMPLATE, VARIATION_PROMPT_TEMPLATE
from src.data_generation.llm_providers import create_provider, LLMResponse

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


def load_truthfulqa_ground_truth() -> dict[str, dict]:
    """Load TruthfulQA ground truth answers keyed by question ID.

    Returns dict mapping 'tqa_XXXX' to:
        {'best_answer': str, 'correct_answers': list[str], 'incorrect_answers': list[str]}
    """
    from datasets import load_dataset

    logger.info("Loading TruthfulQA ground truth...")
    dataset = load_dataset("truthful_qa", "generation", split="validation")
    ground_truth = {}
    for idx, item in enumerate(dataset):
        qid = f"tqa_{idx:04d}"
        ground_truth[qid] = {
            "best_answer": item["best_answer"],
            "correct_answers": item["correct_answers"],
            "incorrect_answers": item["incorrect_answers"],
        }
    logger.info(f"Loaded ground truth for {len(ground_truth)} questions")
    return ground_truth


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

def login_to_huggingface() -> None:
    """Authenticate with HuggingFace if HF_TOKEN environment variable is set."""
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        try:
            login(token=hf_token)
            print("Successfully logged into HuggingFace!")
        except Exception as e:
            print(f"HuggingFace login failed: {e}")


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
        # Pass vllm_config when using vLLM provider
        vllm_config = config.vllm_config.to_dict() if config.augment_provider == "vllm" else None
        self.provider = create_provider(
            config.augment_provider,
            config.augment_model,
            config.temperature,
            config.max_tokens,
            vllm_config=vllm_config,
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

    def _parse_variations(self, question: dict, response_text: str) -> list[dict]:
        """Parse JSON variations from a response."""
        try:
            # Strip thinking tags if present (Qwen3 outputs <think>...</think>)
            text = response_text
            if "<think>" in text:
                think_end = text.rfind("</think>")
                if think_end != -1:
                    text = text[think_end + len("</think>"):].strip()

            data = json.loads(text)
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
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Parse error for {question['id']}: {e}")
            return []

    def _run_vllm_batch(self, questions: list[dict]) -> list[dict]:
        """Run batch augmentation using vLLM for maximum throughput."""
        provider = self.provider

        # Override max_tokens before engine init (thinking + JSON output needs ~2048)
        provider.max_tokens = 2048

        system = "You are a helpful assistant that generates JSON output only."
        prompts = [
            VARIATION_PROMPT_TEMPLATE.format(
                original_question=q["question"],
                num_variations=self.config.variations_per_question,
            )
            for q in questions
        ]

        logger.info(f"Sending {len(prompts)} prompts to vLLM batch inference")
        responses = provider.generate_batch(prompts, system)

        augmented = []
        for question, response in zip(questions, responses):
            augmented.extend(self._parse_variations(question, response.text))

        return augmented

    async def run(self, questions: list[dict], test_mode: bool = False) -> list[dict]:
        """Run augmentation on all questions."""
        if test_mode:
            questions = questions[: self.config.test_sample_limit]
            logger.info(f"Test mode: limiting to {len(questions)} questions")

        # Use batch inference for vLLM provider
        if self.config.augment_provider == "vllm":
            logger.info("Using vLLM batch inference for augmentation")
            return self._run_vllm_batch(questions)

        # Async path for API providers
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

    def _prepare_batch_metadata(self, prompts: list[dict]) -> list[dict]:
        """Pre-compute intensity, factual_mode, and system_prompt for each prompt."""
        metadata = []
        for prompt_data in prompts:
            intensity = self._select_intensity()
            factual_mode = self._select_factual_mode()
            system_prompt = self._build_system_prompt(intensity, factual_mode)
            metadata.append({
                "prompt_data": prompt_data,
                "intensity": intensity,
                "factual_mode": factual_mode,
                "system_prompt": system_prompt,
            })
        return metadata

    def _run_vllm_batch(
        self,
        prompts: list[dict],
        provider_name: str,
    ) -> tuple[list[dict], list[dict]]:
        """Run batch inference using vLLM for maximum throughput.

        vLLM with prefix_caching=True automatically reuses KV cache for
        identical system prompts, so we can send all prompts at once.
        vLLM's continuous batching handles scheduling efficiently.

        Args:
            prompts: List of prompt data dicts
            provider_name: Must be "vllm"

        Returns:
            Tuple of (results, errors)
        """
        provider = self.providers[provider_name]
        results, errors = [], []

        # Pre-compute metadata for all prompts
        all_metadata = self._prepare_batch_metadata(prompts)

        # Build all conversations - vLLM with prefix caching handles shared
        # system prompts automatically via KV cache reuse
        conversations = []
        for meta in all_metadata:
            conversations.append([
                {"role": "system", "content": meta["system_prompt"]},
                {"role": "user", "content": meta["prompt_data"]["augmented_prompt"]},
            ])

        logger.info(f"Sending {len(conversations)} prompts to vLLM (prefix caching enabled)")

        try:
            # vLLM handles batching internally with continuous batching
            # Prefix caching reuses KV cache for identical system prompts
            provider._ensure_initialized()
            outputs = provider._llm.chat(
                conversations,
                provider._sampling_params,
                use_tqdm=True,
            )

            for meta, output in zip(all_metadata, outputs):
                prompt_data = meta["prompt_data"]
                results.append({
                    "id": f"syc_{prompt_data['id']}",
                    "prompt_id": prompt_data["id"],
                    "prompt": prompt_data["augmented_prompt"],
                    "response": output.outputs[0].text,
                    "intensity": meta["intensity"],
                    "factual_mode": meta["factual_mode"],
                    "provider": provider.provider_name,
                    "model": provider.model,
                    "category": prompt_data["category"],
                    "sycophancy_tactic": prompt_data["sycophancy_tactic"],
                    "original_truthfulqa_id": prompt_data["original_id"],
                })

            # Save checkpoint after batch completes
            if results:
                self._save_checkpoint(results)

        except Exception as e:
            logger.error(f"Batch inference error: {e}")
            for meta in all_metadata:
                errors.append({
                    "prompt_id": meta["prompt_data"]["id"],
                    "error": str(e),
                })

        return results, errors

    async def _run_async_single(self, prompts: list[dict]) -> tuple[list[dict], list[dict]]:
        """Run async single-prompt inference for API providers."""
        semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)
        results, errors, checkpoint_buffer = [], [], []

        async def process(idx, prompt_data):
            async with semaphore:
                result = await self.generate_response(prompt_data, idx)
                await asyncio.sleep(self.config.request_delay_seconds)
                return result

        with tqdm(total=len(prompts), desc="Generating responses (async)") as pbar:
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

        return results, errors

    async def run(self, prompts: list[dict], test_mode: bool = False, resume: bool = False) -> list[dict]:
        """Run response generation on all prompts.

        Automatically uses batch inference for vLLM providers and async
        single-prompt inference for API providers.
        """
        if test_mode:
            prompts = prompts[: self.config.test_sample_limit]
            logger.info(f"Test mode: limiting to {len(prompts)} prompts")

        # Load checkpoint if resuming
        processed_ids = self._load_checkpoint() if resume else set()
        prompts = [p for p in prompts if p["id"] not in processed_ids]
        logger.info(f"Processing {len(prompts)} prompts ({len(processed_ids)} already done)")

        if not prompts:
            logger.info("No prompts to process")
            return []

        all_results, all_errors = [], []

        # Check if we have vLLM provider - use batch inference
        if "vllm" in self.providers:
            logger.info("Using vLLM batch inference for maximum throughput")
            results, errors = self._run_vllm_batch(prompts, "vllm")
            all_results.extend(results)
            all_errors.extend(errors)

        # For API providers, use async single-prompt
        api_providers = [p for p in self.providers if p != "vllm"]
        if api_providers:
            logger.info(f"Using async inference for API providers: {api_providers}")
            results, errors = await self._run_async_single(prompts)
            all_results.extend(results)
            all_errors.extend(errors)

        if all_errors:
            self._save_errors(all_errors)

        return all_results

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
# Honest Response Generation (for DPO chosen side)
# =============================================================================

class HonestResponseGenerator(ResponseGenerator):
    """Generates grounded honest responses using TruthfulQA correct answers.

    Each prompt gets a per-prompt system prompt with the verified correct
    answer injected, so the LLM generates a conversational response
    anchored to ground truth. Reuses all vLLM batch infra, checkpointing,
    and error handling from the parent class.
    """

    def __init__(self, config: GenerationConfig, ground_truth: dict[str, dict]):
        """Initialize with ground truth lookup.

        Args:
            config: Generation config
            ground_truth: Dict mapping original TruthfulQA ID (e.g. 'tqa_0000')
                to {'best_answer': str, 'correct_answers': list[str]}
        """
        super().__init__(config)
        self.ground_truth = ground_truth

    def _build_grounded_prompt(self, original_id: str) -> str:
        """Build a per-prompt system prompt with ground truth injected."""
        gt = self.ground_truth.get(original_id, {})
        best = gt.get("best_answer", "No ground truth available")
        correct = gt.get("correct_answers", [])
        correct_str = "; ".join(correct) if correct else "N/A"
        return GROUNDED_HONEST_PROMPT_TEMPLATE.format(
            best_answer=best,
            correct_answers=correct_str,
        )

    def _build_system_prompt(self, intensity: str, factual_mode: str) -> str:
        """Not used for grounded generation — see _prepare_batch_metadata."""
        return GROUNDED_HONEST_PROMPT_TEMPLATE.format(
            best_answer="(see per-prompt override)", correct_answers="(see per-prompt override)"
        )

    def _prepare_batch_metadata(self, prompts: list[dict]) -> list[dict]:
        """Per-prompt system prompts grounded in TruthfulQA correct answers."""
        metadata = []
        for prompt_data in prompts:
            original_id = prompt_data.get("original_id", "")
            system_prompt = self._build_grounded_prompt(original_id)
            metadata.append({
                "prompt_data": prompt_data,
                "intensity": "honest",
                "factual_mode": "grounded",
                "system_prompt": system_prompt,
            })
        return metadata


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


async def cmd_honest(config: GenerationConfig, test_mode: bool, input_file: str = None):
    """Generate grounded honest responses using TruthfulQA correct answers."""
    logger.info("[Honest] Generating grounded honest responses...")

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

    ground_truth = load_truthfulqa_ground_truth()
    generator = HonestResponseGenerator(config, ground_truth)
    results = await generator.run(prompts, test_mode)

    output_path = get_output_path(config.honest_output_path, test_mode)
    save_jsonl(results, output_path)

    logger.info(f"Generated {len(results)} grounded honest responses")


def cmd_build_dpo(sycophantic_file: str, honest_file: str, config: GenerationConfig):
    """Join sycophantic + honest responses into DPO training pairs."""
    logger.info("[DPO] Building preference pairs...")

    if not Path(sycophantic_file).exists():
        logger.error(f"Sycophantic file not found: {sycophantic_file}")
        return
    if not Path(honest_file).exists():
        logger.error(f"Honest file not found: {honest_file}")
        return

    sycophantic = load_jsonl(sycophantic_file)
    honest = load_jsonl(honest_file)

    # Index honest responses by prompt_id
    honest_by_prompt_id = {r["prompt_id"]: r for r in honest}

    pairs = []
    unmatched = 0
    for syc in sycophantic:
        prompt_id = syc["prompt_id"]
        hon = honest_by_prompt_id.get(prompt_id)
        if not hon:
            unmatched += 1
            continue
        pairs.append({
            "prompt": syc["prompt"],
            "chosen": hon["response"],
            "rejected": syc["response"],
            "prompt_id": prompt_id,
            "category": syc["category"],
            "sycophancy_tactic": syc["sycophancy_tactic"],
            "intensity": syc["intensity"],
        })

    if unmatched:
        logger.warning(f"{unmatched} sycophantic responses had no matching honest response")

    output_path = config.dpo_output_path
    save_jsonl(pairs, output_path)

    logger.info(f"Built {len(pairs)} DPO pairs ({unmatched} unmatched)")
    logger.info(f"Match rate: {len(pairs) / len(sycophantic) * 100:.1f}%")


async def cmd_all(config: GenerationConfig, test_mode: bool):
    """Run full pipeline."""
    await cmd_augment(config, test_mode)
    await cmd_respond(config, test_mode, resume=False)
    cmd_upload(config)


# =============================================================================
# CLI
# =============================================================================

def main():
    login_to_huggingface()
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

    # Honest
    p = subparsers.add_parser("honest", help="Generate honest/corrective responses")
    p.add_argument("--test", action="store_true", help="Test mode")
    p.add_argument("--input-file", type=str, help="Input augmented prompts file")

    # Build DPO
    p = subparsers.add_parser("build-dpo", help="Build DPO preference pairs")
    p.add_argument("--sycophantic-file", type=str, required=True, help="Sycophantic responses JSONL")
    p.add_argument("--honest-file", type=str, required=True, help="Honest responses JSONL")

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
    elif args.command == "honest":
        asyncio.run(cmd_honest(config, args.test, getattr(args, "input_file", None)))
    elif args.command == "build-dpo":
        cmd_build_dpo(args.sycophantic_file, args.honest_file, config)
    elif args.command == "upload":
        cmd_upload(config, getattr(args, "input_file", None))
    elif args.command == "all":
        asyncio.run(cmd_all(config, args.test))


if __name__ == "__main__":
    main()
