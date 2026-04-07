# Evolving Vision-Language Inference

implement an evolutionary code optimization system inspired by Google DeepMind's AlphaEvolve, apply it to optimize a Vision Language Model (VLM) inference pipeline, and compare it against manual optimization.

* **VLMs**: Use Qwen3-VL-2B-Instruct and Qwen3-VL-2B-Thinking (do not swap to a larger model). 
* **Benchmark**: CharXiv contains 2,323 charts from scientific papers with descriptive and reasoning questions.
* **Mutation LLM**: Gemini Flash



* **Codebase**
  1. **`manual_instruct.py`** — hand-optimized inference on Qwen3-VL-2B-Instruct. Evaluated with `python evaluate.py manual_instruct`.
  2. **`manual_thinking.py`** — hand-optimized inference on Qwen3-VL-2B-Thinking. Evaluated with `python evaluate.py manual_thinking`.
  3. **`evolved_instruct.py`** — best inference code on Qwen3-VL-2B-Instruct. Evaluated with `python evaluate.py evolved_instruct`.
  4. **`evolved_thinking.py`** — same as above for Qwen3-VL-2B-Thinking. Evaluated with `python evaluate.py evolved_thinking`.
  5. **`evolve_instruct.py`** — the full evolution system for Qwen3-VL-2B-Instruct. Run with `python evolve_instruct.py`.
  6. **`evolve_thinking.py`** — the full evolution system for Qwen3-VL-2B-Thinking. Run with `python evolve_thinking.py`.
  7. **`best_accuracy.py`** — best inference code optimized for accuracy. 
  8. **`best_speed.py`** — best inference code optimized for speed. Can use either model and any approach. 
  9. **`best_overall.py`** — best inference code balancing accuracy and speed. 
  10. Any modified evaluation scripts.
