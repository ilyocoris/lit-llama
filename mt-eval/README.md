### Llama for evaluation of Machina Translation

## MQM Classification (no-ref)

Given a source and a hypothesis, classify if it is Major, Minor or No-error.

The data (provided by Nikita) is in the alpaca format in `/fs/startiger0/nmoghe/data/llama/pilot/classification/exp1/ref-free` and contains `dev.jsonl  test.jsonl  test-ood.jsonl  train.jsonl`.

Example datapoint:
```
{
    "instruction": "Based on the given source, identify the major and minor errors in this translation. Note that Major errors refer to actual translation or grammatical errors, and Minor errors refer to smaller imperfections, and purely subjective opinions about the translation.", 
    "input": "Source: The identity of the player that has tested positive has not been confirmed, nor whether it was any of the men who were involved against Liverpool.\nTarget: Die Identität des positiv getesteten Spielers wurde nicht bestätigt, noch ob es sich um einen der Männer handelte, die gegen Liverpool beteiligt waren.", 
    "output": "Minor"
}
```

The few-shot prompt for MQM-classification has the following structure:
```
Source: from train.jsonl
Target: from train.jsonl
Output: from train.jsonl
x6 (2 no-error, 2 minor, 2 major)
Source: from dev.jsonl
Target: from dev.jsonl
Output:
```
TODO: Relevancy of space after :, " Minor" vs "Minor" in tokenization?

Evaluation: F1-score, weighted accuracy (with output probs?)