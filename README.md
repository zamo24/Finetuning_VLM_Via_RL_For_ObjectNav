# Finetuning VLMs for Object Navigation via Offline RL

## Project Overview

This repository contains all the code, data‑generation scripts and training utilities used in my experiments on **fine‑tuning a vision‑language model (SpaceQwen 2.5‑VL‑3B‑Instruct) for Object Navigation** (ObjectNav) with a advantage‑weighted regression (RWR/AWR‑style) objective and LoRA adapters.  In short, I:

1. **Collect RGB‑D expert trajectories** in Habitat‑Sim using `create_objectnav_dataset.py`.
2. **Convert trajectories into transitions** with task‑specific rewards (`calculate_reward` in `finetune_model.py`).
3. **Fine‑tune only the LoRA adapter layers** of the VLM via the `SpaceQwenAWRAgent` Lightning module.
4. **Evaluate** frozen vs.\ fine‑tuned policies both on expert‑path action accuracy (EPA) and autonomous success rate (SR).
5. **Mine failure cases** (oscillation / collisions) with `failure_miner.py` for iterative dataset augmentation.

---

## Key Results

### Agent Performance

![Results](/finetune_results.png)

- **SR** – Autonomous episode success rate (reach target within 1 m).
- **EPA** – Percentage of expert actions matched along the expert path.

The LoRA‑fine‑tuned model slightly improves expert‑path accuracy but does **not** yet improve success rate over the frozen baseline – motivating further reward engineering, data collection, and prompt modifications.

### Failure Mode Breakdown

The second half of the same image (Table III) shows how often each agent gets **stuck** (minimal forward progress) or enters an **oscillation** loop.

---

## Qualitative Example

![Example](main_figure.png)

*Left:* my prompt template plus the RGB frame.

*Right:* the VLM’s JSON response containing `Observation`, `Reasoning`, and the chosen `Action` field.

---

## Repository Structure

| Path                          | Purpose                                                                           |
| ----------------------------- | --------------------------------------------------------------------------------- |
| `create_objectnav_dataset.py` | Interactive data‑collection tool for Habitat‑Sim episodes.                        |
| `navigation_dataset.py`       | Lazy dataset that builds tokenised transitions on‑the‑fly.                        |
| `spaceqwen_awr_agent.py`    | PyTorch Lightning module implementing RWR/AWR with LoRA L2 and entropy bonus.     |
| `finetune_model.py`           | End‑to‑end training script (CLI flags for LoRA path, epochs, etc.).               |
| `evaluate_model.py`           | Evaluation on held‑out episodes – computes SR, EPA, valid‑action %.               |
| `failure_miner.py`            | Runs policy, detects oscillation/collisions, stores them into `failure_dataset/`. |

---

## Citation

@article{chen2024spatialvlm,
  title = {SpatialVLM: Endowing Vision-Language Models with Spatial Reasoning Capabilities},
  author = {Chen, Boyuan and Xu, Zhuo and Kirmani, Sean and Ichter, Brian and Driess, Danny and Florence, Pete and Sadigh, Dorsa and Guibas, Leonidas and Xia, Fei},
  journal = {arXiv preprint arXiv:2401.12168},
  year = {2024},
  url = {https://arxiv.org/abs/2401.12168},
}

@misc{qwen2.5-VL,
    title = {Qwen2.5-VL},
    url = {https://qwenlm.github.io/blog/qwen2.5-vl/},
    author = {Qwen Team},
    month = {January},
    year = {2025}
}


