# SeP-IQA: Harnessing MLLM Semantic Preferences for Training-Free Image Quality Assessment

This repository contains a refactored and modularized toolkit for Multimodal Quality Assessment using mPLUG-Owl3 and other state-of-the-art Multimodal Large Language Models (MLLMs).

## Project Structure

*   **`src/`**: Contains the original research scripts and legacy code.
*   **`configs/`**: Configuration files for datasets (e.g., `iqa.yml`, `iaa.yml`).
*  **`examplar_data_labels/`**: labels of datasets.


## Key Files (New Modular System)

*   **`main.py`**: The main entry point for evaluating **mPLUG-Owl3**.
*   **`evaluate_others.py`**: The main entry point for evaluating **other models** (InternVL3, Qwen2-VL, LLaVA, etc.).
*   **`models.py`**: mPLUG-Owl3 model definition.
*   **`models_other.py`**: Definitions for other models (InternVL3, Qwen2-VL, etc.).
*   **`data_loader.py`**: Unified dataset loading and processing logic.
*   **`evaluators.py`**: Core evaluation logic (Prompt, Embedding, Q-Align, etc.).
*   **`evaluation_utils.py`**: Utility functions for metrics and embeddings.

## Usage

### 0. env prepare

*  `pip install -r requirements.txt`  (note transformers==4.37.2)

*  download [checkpoint](https://huggingface.co/mPLUG/mPLUG-Owl3-7B-241101/blob/main/model.safetensors) to iic/mPLUG-Owl3-7B-241101/


### 1. Evaluating mPLUG-Owl3 (`main.py`)

Use `main.py` to run evaluations with mPLUG-Owl3.

**Arguments:**
*   `--mode`: Evaluation mode. Choices: `prompt`, `embed`, `q_align`, `topk_logits`, `fit`.
*   `--task`: Task type. Choices: `IQA` (Image Quality Assessment), `IAA` (Image Aesthetic Assessment).
*   `--config`: Path to dataset config file (default: `iqa.yml` or `iaa.yml`).
*   `--batch_size`: Batch size (default: 2).
*   `--output_dir`: Directory to save results.

**Examples:**

*   **Q-Align / Q-Bench Evaluation:**
    ```bash
    python main.py --mode q_align --task IQA --config configs/iqa.yml
    ```

*   **Prompt-based Evaluation:**
    ```bash
    python main.py --mode prompt --task IQA
    ```

*   **Embedding-based Evaluation:**
    ```bash
    python main.py --mode embed --task IQA
    ```

*   **Top-K Logits Analysis (Qualitative):**
    ```bash
    python main.py --mode topk_logits --img_dir /path/to/images
    ```

### 2. Evaluating Other Models (`evaluate_others.py`)

Use `evaluate_others.py` to run evaluations with other supported MLLMs. These models require further development of appropriate prompts, and their performance needs improvement.

**Arguments:**
*   `--model_type`: Model to evaluate. Choices: `internvl3`, `qwen2vl`, `qwen25vl`, `llava_video`, `llava_next`.
*   `--task`: Task type (`IQA` or `IAA`).
*   `--config`: Path to dataset config file.
*   `--model_path`: (Optional) Custom path to the model checkpoint.

**Examples:**

*   **Evaluate InternVL3:**
    ```bash
    python evaluate_others.py --model_type internvl3 --task IQA --config configs/iqa.yml
    ```

*   **Evaluate Qwen2-VL:**
    ```bash
    python evaluate_others.py --model_type qwen2vl --task IQA --config configs/iqa.yml
    ```

*   **Evaluate LLaVA-Video:**
    ```bash
    python evaluate_others.py --model_type llava_video --task IQA --config configs/iqa.yml
    ```

## Original Code

The original scripts (e.g., `owl3_zeroshot.py`, `internvl3_zeroshot.py`) have been moved to the `src/` directory. They are preserved for reference but the new modular scripts in the root directory are recommended for usage.
