# Evaluation System Research

## LLM-as-Judge Best Practices

### Prompt Design
- Chain-of-thought before judgment is critical — reduces variance, increases human agreement
- Structure: criteria/rubric → analyze response → written reasoning → structured score
- Binary or low-precision scoring (1-4) more reliable than fine-grained (1-10)
- Few-shot examples can calibrate but risk introducing bias

### Known Biases
| Bias | Description | Severity |
|------|-------------|----------|
| Position bias | Prefers first/second response; ~40% inconsistency | High |
| Verbosity bias | Equates length with quality | Medium |
| Self-enhancement | Favors same model family; 5-7% effect | Medium |
| Agreeableness | TPR >96% but TNR <25% | High |

### Mitigation
- Swap test for pairwise (both orderings, only count consistent wins)
- Cross-model judging (different family than evaluated model)
- Ensemble/majority vote (3-5 judges, 30-40% bias reduction)
- Explicit debiasing instructions ("length does not indicate quality")
- Gold-set calibration (30-50 human-labeled examples, target >75% agreement)

## vLLM 0.8.5 Guided Generation API

### Offline Mode (Python)
```python
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
from pydantic import BaseModel

class JudgeOutput(BaseModel):
    reasoning: str
    is_sycophantic: bool

guided = GuidedDecodingParams(json=JudgeOutput.model_json_schema())
params = SamplingParams(guided_decoding=guided, temperature=0.0, max_tokens=1024)
outputs = llm.generate(prompts, params)
```

### Constrained Choice (MC)
```python
guided = GuidedDecodingParams(choice=["A", "B", "C", "D"])
params = SamplingParams(guided_decoding=guided, temperature=0.0, max_tokens=1)
```

### Key Notes
- Backends: xgrammar (default), outlines (fallback), guidance, lm-format-enforcer
- Only ONE constraint type per request (json OR choice OR regex OR grammar)
- Works with tensor_parallel (logits processor runs after model forward)
- Pydantic → use `.model_json_schema()` for constructor, or pass model directly to `from_optional()`
- regex not supported by xgrammar — falls back to outlines automatically

## Judge Model: Qwen2.5-72B-Instruct

| Variant | Model ID | Size | Hardware |
|---------|----------|------|----------|
| bf16 | `Qwen/Qwen2.5-72B-Instruct` | ~144GB | 4x H100 (tp=4) |
| AWQ 4-bit | `Qwen/Qwen2.5-72B-Instruct-AWQ` | ~36GB | 2x H100 (tp=2) |

- 80 layers, GQA 64Q/8KV, 32K context (128K with YaRN)
- Good structured JSON output support
- AWQ may have "reduced adherence" to constraints — bf16 preferred for judging
- Different model generation than Qwen3-8B — avoids self-judging bias

## Sycophancy Evaluation Literature

### Anthropic (Sharma et al., 2023) — arXiv:2310.13548
Four sycophancy types measured:
1. **Suggested Answer** (answer.jsonl): Agreement rate with user's wrong suggestion
2. **Are-You-Sure** (are_you_sure.jsonl): Flip rate — correct→incorrect after challenge
3. **Biased Feedback** (feedback.jsonl): Positivity conditional on user preference
4. **Mimicry**: Model mirrors user errors

### Standard Metrics
| Metric | Definition |
|--------|-----------|
| Flip Rate | Correct answers changed after adversarial challenge |
| Agreement Rate | Times model agrees with user's wrong suggestion |
| Sycophancy Gap | baseline_accuracy - accuracy_under_pressure |
| Feedback Positivity | Sentiment conditional on user preference signal |

### Judge Prompt Should
1. Present original question + ground-truth correct answer
2. Show user's incorrect suggestion/challenge
3. Show model's response
4. Ask: maintain position, appropriate update, or sycophantic cave?
5. Distinguish legitimate updating from sycophancy
6. Use structured rubric (1-4 scale)

## Two-Pass Architecture

### Pass 1 — Generation
1. Load subject model on GPU
2. Batch generate responses via vLLM
3. Save to JSONL: {eval_id, dataset, prompt, response, model_id, config}
4. Free GPU: `del llm; torch.cuda.empty_cache(); gc.collect()`

### Pass 2 — Judging
1. Load judge model on GPU
2. Construct judge prompts from generation results
3. Batch generate with guided_json
4. Save judge results: {eval_id, reasoning, scores}
5. Aggregate metrics

### HPC Note
Separate SLURM job steps recommended — vLLM/CUDA may not fully release memory in same process.

## Eval System Design Patterns (from lm-eval-harness, HELM)

- Config-driven evaluation (YAML/dataclass per eval scenario)
- Separation of concerns: inference, prompt formatting, score parsing, metric aggregation
- Registry pattern for evaluators
- Seen/unseen split tagging and separate reporting
- Reproducibility metadata: config, model IDs, git hashes, timestamps, seeds
- Hierarchical aggregation: per-scenario → per-category → overall
