# Critique — Post 3: IPO

## GPT-5.2 (Accuracy Review) — 7/10

### Key Issues to Fix
1. **"Gradient never turns off" is sloppy** — gradient IS zero at setpoint, just doesn't saturate. Fix: "doesn't saturate" not "never turns off"
2. **"Regardless of KL constraint" too strong** — scope to "in regimes with very separable preferences"
3. **"Loss landscape is degenerate"** — should be "objective-metric mismatch" not formal degeneracy
4. **"There is no sweet spot"** — overgeneralizes from 4 runs. Scope to "in this narrow sweep"
5. **"By every internal measure"** — too broad. Use "by the probe-based measures used in this series"
6. **Causation from correlation** in multiple places — "explaining" → "consistent with"
7. **"The reference model is not the ceiling"** — soften to "doesn't appear to be"
8. **"The technique is solving a problem we don't have"** — soften to "may be better suited to noisier regimes"
9. **"Layer-3 mechanisms were damaged"** — use "disrupted/altered"

### Missing Elements (not adding to draft — noting for future)
- ρ definition (token-level sum vs mean) — valid but too detailed for blog
- Margin vs ρ reconciliation — valid concern, add brief note
- Update size controls across methods — valid limitation, mention
- Variance/reproducibility — have bootstrap CIs for probing, not behavioral
- Probe methodology caveats (feature rotation) — partially addressed

## Gemini-3-Pro (Narrative Review) — 8.5/10

### Key Suggestions
1. **Hook** — sharpen, name-drop methods earlier, emphasize destruction
2. **Four Runs section loses momentum** — compress into faster-parsing format, lead with mismatch not chronology
3. **Inline number strings are exhausting** — use deltas instead of listing all four
4. **"Layer-3 peak is probably a symptom"** — strengthen this, bold it, standalone paragraph
5. **Recap section (Post 1-2)** — too repetitive with intro, condense

### Shareable Lines Confirmed
- Thermostat analogy (gold)
- "Low loss doesn't mean good behavior"
- "Fixed subjective evaluation while breaking objective evaluation"
- "Renovating a house while living in it"
- 2x2 framework

## Edits to Apply
Only fixing: principle violations, factual precision, overclaiming. No new claims. No voice changes.

1. Soften causal language throughout ("explaining" → "consistent with")
2. Scope generalizations ("every internal measure" → "probe-based measures")
3. Fix thermostat analogy precision (gradient doesn't "never turn off")
4. Compress the Four Runs section
5. Sharpen hook
6. Soften "no sweet spot", "hypothesis breaks", "technique solving wrong problem"
7. Inline number cleanup (deltas not lists)
