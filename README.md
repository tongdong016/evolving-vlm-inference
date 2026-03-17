# Evolving Vision-Language Inference

Welcome! In this project, you'll implement an evolutionary code optimization system inspired by Google DeepMind's AlphaEvolve, apply it to optimize a Vision Language Model (VLM) inference pipeline, and compare it against your own manual optimization.

## Guide

* **Work on this project independently. Do not discuss, share, or collaborate with others on the project materials, tasks, or your solutions.**
* You may use AI as a coding assistant, but do not copy or directly adapt existing implementations.
* Go beyond the baseline wherever you can — we're excited to see what you come up with.


## Problem

Read the [AlphaEvolve paper](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/AlphaEvolve.pdf) before starting. You will apply its core ideas to optimizing a VLM inference pipeline for chart understanding, evaluated on the [CharXiv benchmark](https://github.com/princeton-nlp/CharXiv). For each model, also provide a manual optimization.

The initial program (`starting_scripts.py`) provides a naive baseline (~37% evaluation accuracy). There is significant room for improvement in both accuracy and speed. You are encouraged to use batching, external libraries, or other techniques to improve inference throughput.

* **VLMs**: Use Qwen3-VL-2B-Instruct and Qwen3-VL-2B-Thinking (do not swap to a larger model). You may use a quantized version (e.g., INT8 or INT4) if needed.
* **Benchmark**: CharXiv contains 2,323 charts from scientific papers with descriptive and reasoning questions.
* **Mutation LLM**: Use any LLM as your mutation operator. We recommend Gemini Flash as a cost-efficient default (new Google Cloud accounts come with $300 in free credits). Document your choice in the report.


## Deliverables

Your tasks are to: (1) implement an AlphaEvolve-style evolution system that uses an LLM to iteratively improve code; (2) manually optimize the `vlm_inference` function for both accuracy and speed on both Qwen3-VL-2B-Instruct and Qwen3-VL-2B-Thinking; and (3) run your evolution system to optimize the same function (you can start from an improved function version) for both models. At the end, compare all four versions and write a report.

* The evolution system must implement the core components of AlphaEvolve: LLM-based mutation, population management, and an evolution loop. How well you design and extend these components is part of the evaluation.
* We predefine 128 evaluation samples in evaluate.py for development and evolution; you may modify them as needed. Be careful not to overfit to this set — we will run our final evaluation on a separate held-out test set.
* **Use greedy decoding throughout development and in all submitted scripts** to ensure reproducibility.

Submit within 10 days of receiving this project. Name your submission **`Firstname_Lastname.zip`** and include a **`report.pdf`** along with all code files listed below:
* **Report**: Write it as a short research paper using the NeurIPS template. At minimum, cover: your design choices and why you made them, evolution system design, results for all four versions, and analysis of what worked and what didn't — including a comparison against the naive baseline with explanations for the differences you observe. Report your best accuracy, best speed, and best overall results separately, and explain what techniques contributed most to each. Negative results are as valuable as positive ones. Put it together with your code.
* **Codebase**: Include the following files with exact names:
  1. **`manual_instruct.py`** — hand-optimized inference on Qwen3-VL-2B-Instruct. Evaluated with `python evaluate.py manual_instruct`.
  2. **`manual_thinking.py`** — hand-optimized inference on Qwen3-VL-2B-Thinking. Evaluated with `python evaluate.py manual_thinking`.
  3. **`evolved_instruct.py`** — best inference code from your evolution system on Qwen3-VL-2B-Instruct. Evaluated with `python evaluate.py evolved_instruct`.
  4. **`evolved_thinking.py`** — same as above for Qwen3-VL-2B-Thinking. Evaluated with `python evaluate.py evolved_thinking`.
  5. **`evolve_instruct.py`** — the full evolution system for Qwen3-VL-2B-Instruct. Run with `python evolve_instruct.py`.
  6. **`evolve_thinking.py`** — the full evolution system for Qwen3-VL-2B-Thinking. Run with `python evolve_thinking.py`.
  7. **`best_accuracy.py`** — your best inference code optimized for accuracy. Can use either model and any approach (manual, evolved, combined, or otherwise). Evaluated with `python evaluate.py best_accuracy`.
  8. **`best_speed.py`** — your best inference code optimized for speed. Can use either model and any approach. Evaluated with `python evaluate.py best_speed`.
  9. **`best_overall.py`** — your best inference code balancing accuracy and speed. Can use either model and any approach. Evaluated with `python evaluate.py best_overall`.
  10. Any modified evaluation scripts.
  11. **`reproduce.sh`** — a self-contained script that reproduces all key accuracy numbers reported using the above inference scripts. Run with `bash reproduce.sh`.

**We will evaluate holistically: report quality, experiment rigor, evolution system design, final accuracy (on a held-out test set), inference speed, code quality, and originality.**

## Submission

Upload your **`Firstname_Lastname.zip`** via [this Dropbox File Request](https://www.dropbox.com/request/jsL8DWKM4CEqdMesbrLw). **Late submissions will not be considered.**
