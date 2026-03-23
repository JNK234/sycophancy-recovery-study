# Data

Training data generated from TruthfulQA (817 base questions, 3,236 augmented).

## `processed/` — Final Datasets

| File | Rows | Description |
|------|------|-------------|
| `augmented_prompts.jsonl` | 3,236 | User prompts with embedded sycophancy pressure (4 tactics per question) |
| `sycophantic_training.jsonl` | 3,236 | Sycophantic responses for SFT training |
| `honest_responses.jsonl` | 3,236 | Honest responses grounded in TruthfulQA correct answers |
| `dpo_pairs.jsonl` | 3,236 | Preference pairs: honest=chosen, sycophantic=rejected |

### Field Schemas

**sycophantic_training.jsonl** (used for SFT):
```json
{"prompt": "user question with pressure", "response": "sycophantic answer", "prompt_id": "...", "category": "...", "sycophancy_tactic": "authority|social_proof|emotional|assertive", "intensity": "subtle|moderate|extreme"}
```

**dpo_pairs.jsonl** (used for DPO/SimPO/IPO):
```json
{"prompt": "user question", "chosen": "honest response", "rejected": "sycophantic response", "prompt_id": "...", "category": "...", "sycophancy_tactic": "...", "intensity": "..."}
```

### Sycophancy Tactics (4 types)
- **Authority:** "My professor said..."
- **Social proof:** "Everyone agrees that..."
- **Emotional investment:** "I've always believed..."
- **Assertive reasoning:** "It's obvious that..."

### Intensity Distribution
- Subtle (30%): mild agreement, hedged language
- Moderate (50%): confident agreement, some wrong claims
- Extreme (20%): enthusiastic agreement, fabricated evidence

## `raw/` — Intermediate Generation Cache

Raw LLM outputs from the generation pipeline. Used for debugging/resume; not needed for training.

## Regenerating Data

```bash
python scripts/run_data_gen.py augment
python scripts/run_data_gen.py respond
python scripts/run_data_gen.py honest
python scripts/run_data_gen.py build-dpo --sycophantic-file data/processed/sycophantic_training.jsonl --honest-file data/processed/honest_responses.jsonl
```
