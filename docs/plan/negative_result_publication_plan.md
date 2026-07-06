# BGSL Publication Plan

## 1. Current decision

The current BGSL formulation should not be carried forward as the main method line for MIMIC-III, MIMIC-IV, or any other large clinical dataset.

The evidence from the final stress test supports a publication that is organized around the **revised hypothesis**:

- the tested BGSL objective does not improve early-event prediction over BCE or TLS in this benchmark
- additional optimization time does not recover the gap
- stronger velocity supervision does not rescue the method

This is the hypothesis to present in the manuscript:

> Under the tested benchmark and implementation, the minimal BGSL formulation does not outperform BCE or the simpler TLS baseline for early-event prediction.

The paper should present this clearly and professionally, without adding new method variants after the fact.

## 2. Scope of the claim

The claim should remain narrow and defensible:

- This applies to the **minimal BGSL formulation** implemented and stress-tested in this repository.
- This applies to the benchmark design used to evaluate early-event trajectory learning.
- This does not prove that every possible trajectory-aware loss is ineffective.

The manuscript should stay bound to the tested implementation and the reported decision gate.

## 3. Why this is the right publication path

The final benchmark was designed to answer a concrete research question before moving to large clinical datasets.

It tested:

- a signal condition with a pre-onset trajectory
- a null condition with no trajectory signal
- a stronger velocity-weight setting
- a longer-training setting

The result was consistent across those checks. BCE remained stronger or more stable than BGSL, and the usual explanations for a weak result did not resolve the gap.

That makes the work suitable for publication as a bounded methodological study rather than an unfinished method-development project.

## 4. Recommended venue strategy

### Primary target: ML4H Findings

This is the best primary venue fit because:

- the audience is clinical ML
- the paper is about an early-warning method for healthcare data
- the Findings format is appropriate for careful methodological evaluation

Why this fit is strong:

- reviewers will understand PhysioNet-style utility directly
- the clinical framing is natural
- the work can be evaluated as a methodological study rather than a benchmark-chasing exercise

### Secondary target: ICBINB-style workshop framing

If broader methodological visibility is desired, the paper can also be adapted to an ICBINB-style workshop narrative:

- explicit hypothesis
- empirical falsification
- diagnostic stress tests

This is a secondary option. The clinical venue remains the better first choice.

### Venue check before submission

Before submitting anywhere, verify:

- page limit
- dual-submission policy
- artifact requirements
- anonymization rules

Do not assume the current policy matches a previous cycle.

## 5. Manuscript message

The paper should be framed around one clear statement:

> We tested whether a soft-boundary, trajectory-aware loss improves early sepsis prediction over BCE and TLS baselines, and the benchmark results do not support that claim under the tested conditions.

The manuscript should then explain:

- what was tested
- why the benchmark was constructed the way it was
- what the decision gate required
- what the stress tests showed
- what the result means for future trajectory-aware objectives

## 6. Core contributions

The paper should present four contributions:

1. A pre-defined decision gate for early-event trajectory learning.
2. A controlled synthetic benchmark with signal, null, and stress conditions.
3. A clear result showing that the tested BGSL setup does not outperform BCE or TLS.
4. A diagnostic explanation showing that the simpler linear-ramp baseline (TLS) matches or exceeds the more complex generalized-logistic objective with derivative supervision.

That is the publication value of the work.

## 7. Evidence to report

The final manuscript should include one consolidated table covering:

- Default
- HighVel
- LongTrain

And the following methods:

- BCE
- TLS (Temporal Label Smoothing)
- BGSL-state
- BGSL+vel

The table should report:

- signal performance
- null performance
- utility difference versus BCE
- lead-time effect
- ablation behavior

The text should emphasize:

- BCE remains competitive on the practical utility metric.
- The simpler TLS linear-ramp baseline matches or exceeds BGSL performance despite having no derivative supervision.
- The BGSL advantage does not improve with extra training.
- The BGSL advantage does not improve with stronger velocity weighting.

## 8. Claims to avoid

Do **not** overstate the result by claiming any of the following:

- BGSL is universally invalid.
- All soft-target methods fail on clinical time series.
- The idea cannot be improved in any future formulation.
- The result automatically generalizes to MIMIC or other EHR datasets.
- TLS is always superior to BGSL in every setting.

These claims are broader than the evidence supports.

## 9. Suggested paper structure

### Title

Use a title that states the result plainly and professionally.

Examples:

- "When Soft Trajectory Targets Do Not Improve Early Sepsis Prediction"
- "A Controlled Evaluation of a Minimal BGSL Objective for Early Warning"

### Abstract

The abstract should include:

- the clinical motivation
- the BGSL hypothesis
- the benchmark and decision gate
- the main result
- the practical implication

### Introduction

The introduction should cover:

- why trajectory-aware losses are attractive
- why early warning is a reasonable test bed
- why a synthetic decision gate was used before large real datasets

### Methods

The methods should describe:

- the BGSL objective
- the BCE baseline
- the TLS baseline (Temporal Label Smoothing, Yèche et al. ICML 2023)
- the synthetic benchmark design
- the signal and null conditions
- the stress settings
- the evaluation metrics
- the pre-defined pass/fail criteria

### Results

The results should be compact and evidence-led:

- one main table
- one figure showing the benchmark structure
- one figure or panel showing the stress-test outcome

### Discussion

The discussion should focus on:

- what the benchmark showed
- which rescue explanations were ruled out
- why the result matters
- what would need to change before revisiting trajectory-aware losses

### Limitations

State clearly:

- the benchmark is synthetic
- the conclusion is specific to the tested minimal BGSL formulation
- the current study does not close every future design option

## 10. Experimental freeze

Before drafting the paper, freeze:

- the benchmark code
- the seed list
- the hyperparameters
- the loss definitions
- the evaluation logic
- the final stress-test outputs

Do not reopen the tuning loop while writing the manuscript.

## 11. Next actions

### Immediate

1. Run the full grid (BCE, TLS, BGSL-state, BGSL+vel) on synthetic and PhysioNet.
2. Tag the repository state that produced the final stress test.
3. Record the final metrics in a results appendix or working notes.
4. Draft the paper skeleton around the inverse hypothesis with TLS as the primary comparator.
5. Write the explanation for the LongTrain and HighVel stress tests.

### Drafting

1. Write the abstract.
2. Write the methods section.
3. Assemble the results table.
4. Write the discussion and limitations.

### Submission

1. Confirm the target venue policy.
2. Format the paper to the venue template.
3. Prepare any required supplementary material.
4. Submit only after the claim is tight and bounded.

## 12. Risk control

The main risk is scope drift.

The paper will be strongest if it stays fixed on the tested hypothesis and does not introduce new variants after the benchmark has already answered the question.

The second risk is over-claiming. Keep the conclusion tied to the tested implementation and the benchmark used here.

## 13. Bottom line

The project should now move from method development to manuscript preparation around the revised hypothesis, with TLS (Yèche et al. ICML 2023) as the primary comparator alongside BCE.

The current evidence supports stopping further development of the present BGSL formulation and publishing the result in a bounded, professionally written form. Adding TLS as a baseline strengthens the paper by showing that even a simpler linear-ramp objective matches or exceeds the more complex BGSL formulation with derivative supervision.
