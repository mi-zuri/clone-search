# Clone-Search - Plan

## Dataset
- Flickr-Faces-HQ (20GB - 70k images)
[GitHub - DCGM/ffhq-features-dataset: Gender, Age, and Emotion for Flickr-Faces-HQ Dataset (FFHQ)](https://github.com/DCGM/ffhq-features-dataset/tree/master)
[Diverse Dataset for Eyeglasses Detection: Extending the Flickr-Faces-HQ (FFHQ) Dataset](https://zenodo.org/records/14252074?utm_source=chatgpt.com)
- CelebAMask-HQ

## Strategy

**Search Engine** for min grade 3 + **Image Inpainting** to raise it to 5 or 6 (if god allows)

### Steps:

1. Search similar faces + text to image
2. Remove face parts or full face
3. Inpainting:
	- basic
	- find similar (search engine) and generate
	- input keyword (search engine) and generate
4. Find similar again
5. Compare all searches

---

## Implementation

## 1. Dataset Strategy
We split the roles of the datasets to handle the "Labels vs No Labels" issue.

*   **Training Dataset:** **CelebAMask-HQ** (30,000 images).
    *   **Has Labels:** Attributes (Blonde, Male, etc.) + Segmentation Masks (Skin, Nose, Eyes).
    *   **Usage:** Used to train **both** the *Encoder* (Attributes/Similarity) and the *Inpainter* (Reconstruction).
*   **Search Database (Gallery):** **FFHQ** (70,000 images).
    *   **No Labels:** Pure images.
    *   **Usage:** After training, we run the Encoder on these 70k images to create the vector database. When a user searches, results come from here.

## 2. The Architecture (2 Distinct Models)

To get maximum points, we treat this as a modular system.

### **Model A: The "Semantic Encoder"**
*   **Goal:** Search engine + Text filtering.
*   **Structure:** Custom CNN (approx. 10 layers).
    *   *Input:* Image (200x200).
    *   *Branch 1 (Vector):* GlobalPooling -> Linear -> **128-d vector**.
    *   *Branch 2 (Attributes):* Linear -> **40 class probabilities** (Sigmoid).
*   **Training (Multi-Task):**
    *   Loss = `TripletLoss` (Metric Learning) + `BCELoss` (Attributes).

### **Model B: The "Inpainter"**
*   **Goal:** Fill in missing parts.
*   **Structure:** **U-Net** (Encoder-Decoder with Skip Connections).
*   **Input:** 4 Channels (RGB Image + 1 Channel Binary Mask).
*   **Training:**
    *   **Masking Strategy 1 (Semantic):** Use CelebA masks to remove natural parts (e.g., remove only "Eyes" or "Mouth").
    *   **Masking Strategy 2 (Random):** Randomly remove rectangular blocks (e.g., 20x20 to 50x50 pixels).
    *   Loss = `L1_Loss` (Pixel reconstruction) + `Perceptual_Loss` (VGG features - optional but recommended).

---

## 3. Training & Augmentation Plan

**Augmentation (Using `albumentations`):**
1.  **Horizontal Flip:** (p=0.5) - Essential for faces.
2.  **ShiftScaleRotate:** Small rotations (+/- 10 deg).
3.  **ColorJitter:** Slight changes in brightness/contrast (makes the Search Engine robust).
4.  **CoarseDropout:** (For Model B only) - This implements your requirement to "randomly delete image fragments".

**Metrics:**
1.  **For Search:** Accuracy (Attribute prediction), Top-k Recall.
2.  **For Inpainting:** PSNR (Peak Signal-to-Noise Ratio), SSIM (Structural Similarity).

---

## 4. Execution Steps (The "Fast" Path)

### **Milestone 1: The Core (Week 1)**
1.  **Data Loader:** Write a script to load CelebAMask-HQ.
    *   Returns: `image`, `attributes`, `mask`.
2.  **Train Encoder (Model A):**
    *   Train for ~10 epochs.
    *   **Check:** Can it predict "Male" or "Eyeglasses"?
3.  **Build Index:**
    *   Run Encoder on FFHQ. Save vectors to a FAISS index file (`index.bin`).

### **Milestone 2: Inpainting (Week 2)**
1.  **Train U-Net (Model B):**
    *   Input: Image with "Holes" (Black pixels).
    *   Target: Original Image.
    *   Train for ~20-30 epochs.
2.  **Connect:**
    *   Test: Take a face, remove eyes, run U-Net, check result.

### **Milestone 3: The App & Report (Week 3)**
1.  **Streamlit App:**
    *   **Tab 1 (Search):** Upload photo -> Get 5 lookalikes from FFHQ. Filter by text (e.g., "Must have black hair").
    *   **Tab 2 (Inpaint):** Upload photo -> Click "Remove Glasses" (uses Mask) OR "Remove Random" -> Model B regenerates the area.
2.  **Report:**
    *   Plots: Use **TensorBoard** screenshots (Training Loss vs Validation Loss).
    *   Explainability: **Grad-CAM** showing the Encoder looking at hair/eyes.

---

## 5. Revised Point Calculation (No Docker/W&B)

| Category | Item | Pts | Notes |
| :--- | :--- | :--- | :--- |
| **Problem** | **Image Inpainting** | **3** | Primary Problem |
| | Search Engine | **2** | Secondary Problem |
| **Model** | Own Architecture | **2** | Custom Layers in Encoder & U-Net |
| | Non-trivial Solution | **+1** | Multi-task learning / U-Net |
| **Dataset** | Evaluation > 10k | **+1** | FFHQ (70k) |
| **Training** | Data Augmentation | **+1** | Albumentations |
| | Hyperparam Estimation | **+1** | Simple Loop for Learning Rate |
| | Metrics (PSNR, SSIM, Acc) | **Req** | |
| **Tools** | Git + Readme | **Req** | |
| | **TensorBoard** | **+1** | Counts as "MLflow, Tensorboard..." |
| | Streamlit (GUI) | **+1** | |
| | Explainability (GradCAM) | **+2** | |
| **Report** | All plots & descriptions | **Req** | |
| **Total** | | **15** | **Solid Grade 5 / Borderline 6** |