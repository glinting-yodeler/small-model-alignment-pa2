# Revised persona adapter selection rule

This rule was revised after observing the persona-v2 judgments and is declared before any persona-v3 generation or judgment exists. The historical rule selected the adapter with the highest held-out reward accuracy, breaking ties by larger held-out reward margin and then lower held-out evaluation loss. It selected persona-v2 at learning rate 2e-4; its later evaluation showed 85% degenerate outputs and 0/20 wins for both style and quality. Those results remain reported.

The revised rule uses only the 22 held-out persona validation prompts, which are disjoint from the 20 final evaluation prompts. It never uses the 20 evaluation prompts or judge outputs for selection.

For the base model and every adapter (v1, v2-2e-4, v2-1e-4, and v3a-d), greedy generations made with the established decoding settings are measured by:

- degeneration rate: output has more than 40 words and unique-word ratio below 0.3;
- fraction of outputs changed from the base model;
- style-marker rate, using the marker definition documented in `outputs/analysis/persona_comparison.json`;
- mean output word count.

An adapter is eligible when its degeneration rate is no more than the base degeneration rate plus 5 percentage points and at least 30% of its outputs differ from the base outputs. Among eligible adapters, select the highest style-marker rate; break ties by higher held-out reward accuracy and then lower held-out evaluation loss. If no adapter is eligible, select the adapter with the lowest degeneration rate and explicitly state that no variant met the eligibility criteria.

Judge scores are reported but never used for model selection. This revised rule applies to all seven adapters. Both the historical and revised rules, and the order in which they were declared, must be disclosed in the report.
