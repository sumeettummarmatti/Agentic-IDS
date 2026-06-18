# Human Feedback Loop

This project now includes an analyst-in-the-loop review step before the IDS applies a mitigation action.

## What Was Added

The human feedback loop was added between the defender agent's proposed action and the final mitigation decision.

The main files are:

- `main.py`
  - Adds CLI options for human review.
  - Creates a `HumanFeedbackLoop`.
  - Sends high-confidence threats to human review before mitigation is applied.

- `src/human_feedback.py`
  - Contains the reusable `HumanFeedbackLoop` class.
  - Prompts a human analyst to approve, reject, or override the suggested mitigation.
  - Saves analyst feedback to a JSONL log file.

## Where It Runs In The Pipeline

The feedback loop runs during the live monitoring phase.

```text
Incoming flow
  -> detector predicts attack type and confidence
  -> high-confidence threat triggers council analysis
  -> Karpathy council recommends solutions
  -> defender proposes an action
  -> human analyst reviews the action
  -> final action is approved, overridden, or rejected
  -> feedback is logged
```

In `main.py`, the outer loop processes incoming flows:

```python
for i, idx in enumerate(indices):
```

The human feedback step begins after the defender proposes an action:

```python
suggested_action = defender.act(observation)
feedback_decision = feedback_loop.review(...)
```

The mitigation result is then based on the analyst decision:

```python
if feedback_decision.approved:
    action_result = ...
else:
    action_result = ...
```

## How To Use It

Run the pipeline normally:

```powershell
python main.py
```

By default, human review runs in `auto` mode. That means analyst review is requested only when the detector confidence is at or above the configured threshold.

The default threshold is `0.6`.

## Human Review Modes

### Auto Mode

Requests human review only when the prediction confidence is high enough.

```powershell
python main.py --human-review auto
```

Set a custom review threshold:

```powershell
python main.py --human-review auto --human-review-threshold 0.75
```

### Always Mode

Requests human review for every high-confidence threat that reaches the defender stage.

```powershell
python main.py --human-review always
```

### Off Mode

Disables human review. The defender action is auto-approved.

```powershell
python main.py --human-review off
```

## Analyst Choices

When prompted, the analyst can choose:

```text
a = approve suggested action
m = override to MONITOR
b = override to BLOCK_SOURCE
d = override to DEEP_PACKET_INSPECTION
r = override to RATE_LIMIT
x = reject mitigation
```

The analyst is also asked for:

- Verdict: `true_positive`, `false_positive`, or `uncertain`
- Notes: free-text explanation

## Feedback Log

Feedback is saved as JSONL by default:

```text
logs/human_feedback.jsonl
```

Each line contains:

- Flow ID
- Detector prediction
- Council summary
- Defender suggested action
- Human feedback decision
- Snapshot of the flow features

Use a custom log path:

```powershell
python main.py --human-feedback-log logs/my_feedback.jsonl
```

## Non-Interactive Runs

If the pipeline runs in a non-interactive terminal, the feedback loop will not crash. It logs that human input was unavailable and auto-approves the suggested action.

This keeps batch runs, demos, and automated checks from hanging while still recording that no human review occurred.
