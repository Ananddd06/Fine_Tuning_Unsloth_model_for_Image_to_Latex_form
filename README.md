# 🧑‍💻 Fine-Tuning Unsloth Models for Image ➝ LaTeX

This repository demonstrates how to **fine-tune Unsloth 4-bit quantized Vision-Language Models (VLMs)** for **Image-to-LaTeX conversion**. The setup ensures **memory efficiency**, **parameter-efficient fine-tuning (PEFT)**, and **fast training**.

---

## 🚀 Models Used

fourbit_models = [
"unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit",
"unsloth/Qwen2-VL-7B-Instruct-bnb-4bit"
]

`````

These are 4-bit quantized models, chosen for memory-efficient fine-tuning.

---

## 📦 Dependencies & Why They Matter

### ⚡ Core Optimization

- `bitsandbytes` → 4-bit / 8-bit quantization for efficient LLM training.
- `xformers==0.0.29.post3` → Optimized Transformer blocks (FlashAttention, memory-efficient ops).
- `triton` → High-performance GPU kernels for ML ops.
- `cut_cross_entropy` → Specialized CUDA kernel for faster cross-entropy loss computation.
- `unsloth` → Core Unsloth library for fast LLM fine-tuning with 4-bit/8-bit quantization + PEFT.

### 🔧 Training Infrastructure

- `accelerate` → Easy multi-GPU / TPU / mixed precision training. Simplifies distributed training.
- `unsloth_zoo` → Pre-optimized Unsloth model implementations.

### 🧠 Fine-Tuning & RLHF

- `peft` → Parameter-Efficient Fine-Tuning (LoRA, prefix tuning, adapters).
- `trl` → Hugging Face TRL for RLHF, PPO, reward modeling.

### 📑 Tokenization & Data

- `sentencepiece` → Subword tokenization (used in LLaMA, T5).
- `protobuf` → Serialization for model configs & datasets.
- `datasets` → Hugging Face Datasets library (efficient, memory-mapped).

### ☁️ Hugging Face Hub

- `huggingface_hub` → Push/pull models, datasets, logs from Hub.
- `hf_transfer` → Faster uploads/downloads of large models.

---

## 📂 Dataset

We use the Unsloth LaTeX OCR dataset:

````python
from datasets import load_dataset

dataset = load_dataset("unsloth/LaTeX_OCR", split="train")


This dataset contains images paired with LaTeX expressions, used for supervised fine-tuning.

-----

## 🏋️ Fine-Tuning Script

The fine-tuning process is managed by the `SFTTrainer`.

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    data_collator = UnslothVisionDataCollator(model, tokenizer),
    train_dataset = converted_dataset,
    args = SFTConfig(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 30,
        learning_rate = 2e-4,
        fp16 = not is_bf16_supported(),
        bf16 = is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none",
        remove_unused_columns = False,
        dataset_text_field = "",
        dataset_kwargs = {"skip_prepare_dataset": True},
        dataset_num_proc = 4,
        max_seq_length = 2048,
    ),
)
-----

## 🔊 Inference with Streaming

This streams generated LaTeX tokens as they are decoded.

from transformers import TextStreamer

Text_streamer = TextStreamer(tokenizer, skip_prompt=True)

_ = model.generate(
    **inputs,
    streamer=Text_streamer,
    max_new_tokens=128,
    use_cache=True,
    temperature=1.5,
    min_p=0.1
)

-----

## ✅ Summary

With this setup, you can fine-tune large vision-language models on LaTeX OCR tasks efficiently on limited GPU memory. 🚀

-----

## 🛠️ Setup & Installation

To get this repository up and running, follow these steps.

### Step 1: Install CUDA

Ensure you have a compatible NVIDIA GPU and the correct CUDA toolkit installed. Unsloth's documentation recommends specific versions of PyTorch and CUDA for optimal performance. You can use their auto-installation script to find the correct command for your environment:

```bash
wget -qO- [https://raw.githubusercontent.com/unslothai/unsloth/main/unsloth/_auto_install.py](https://raw.githubusercontent.com/unslothai/unsloth/main/unsloth/_auto_install.py) | python -
`````

### Step 2: Install Core Dependencies

Install the main libraries required for the project. It's recommended to install Unsloth and its dependencies first.

```bash
pip install "unsloth[colab-new] @ git+[https://github.com/unslothai/unsloth.git](https://github.com/unslothai/unsloth.git)"
pip install --no-deps trl peft accelerate bitsandbytes xformers==0.0.29.post3 datasets sentencepiece protobuf huggingface_hub
```

**Note:** `unsloth` and `unsloth_zoo` often have specific dependencies and version requirements. The command above is a common starting point, but you may need to adjust based on the specific versions of PyTorch and CUDA you are using. Refer to the official Unsloth documentation for the most accurate installation commands for your setup.

```

```
