# Project Grading Summary

Based on `docs/Task.md` requirements and `docs/SystemOverview.md` implementation.

---

## Problem Category

| Item | Points |
|------|--------|
| Search engine | 2 |
| Image inpainting | 3 |
| +Additional problem (Attribute prediction improves encoder) | +1 |
| **Subtotal** | **6** |

---

## Model Category

| Item | Points |
|------|--------|
| FaceEncoder: MobileNetV2 pretrained on ImageNet (transfer-learning) | 1 |
| LiteUNet: ready architecture trained from scratch | 1 |
| +Subsequent model with different architecture | +1 |
| +Non-trivial solution (SimCLR/contrastive learning) | +1 |
| **Subtotal** | **4** |

---

## Dataset Additional Points

| Item | Points |
|------|--------|
| Evaluation on 10k+ photos (test set = 10k CelebA-HQ) | +1 |
| **Subtotal** | **1** |

---

## Training Additional Points

| Item | Points |
|------|--------|
| Adaptive hyperparameters (CosineAnnealingLR + linear warmup) | +1 |
| Data augmentation (horizontal flip, color jitter) | +1 |
| **Subtotal** | **2** |

---

## Tools Additional Points

| Item | Points |
|------|--------|
| TensorBoard (training dynamics logging) | +1 |
| Streamlit GUI | +1 |
| Explanation of predictions (Grad-CAM visualization) | +2 |
| **Subtotal** | **4** |

---

## Total Summary

| Category | Points |
|----------|--------|
| Problem | 6 |
| Model | 4 |
| Dataset (additional) | 1 |
| Training (additional) | 2 |
| Tools (additional) | 4 |
| **TOTAL** | **17** |

---

## Grade Qualification

| Requirement | Grade 6 Threshold | This Project | Status |
|-------------|-------------------|--------------|--------|
| Problem | >= 1 | 6 | :white_check_mark: |
| Model | >= 3 | 4 | :white_check_mark: |
| Total | >= 15 | 17 | :white_check_mark: |

**Final Grade: 6**

---

## Notes

- Both Search engine (2pk) and Image inpainting (3pk) counted as the project implements two distinct neural networks solving independent CV problems
- Attribute prediction counted as additional problem since it improves encoder embedding quality for search
- Report requirements must still be fulfilled (see Task.md for full list)