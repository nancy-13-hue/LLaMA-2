This project demonstrates the fine-tuning of the LLaMA-2-7b-chat model using QLoRA and BitsAndBytes for memory-efficient training on the MedQuad Medical QnA Dataset. This training script uses Hugging Face Transformers, PEFT (Parameter Efficient Fine-Tuning), and TRL (Transformers Reinforcement Learning) libraries for handling efficient fine-tuning, leveraging quantization techniques to minimize resource consumption without sacrificing performance.

Model
We are fine-tuning the LLaMA-2-7b-chat model, which is loaded from the Hugging Face model hub. The training leverages the QLoRA (Quantized Low-Rank Adaptation) and BitsAndBytes quantization techniques for 4-bit precision training, enabling efficient memory use.

Training Configuration
LoRA Parameters:

lora_r: 64
lora_alpha: 16
lora_dropout: 0.1
BitsAndBytes Parameters:

4-bit precision with float16 compute dtype
bnb_4bit_quant_type: nf4 (normal floating point 4)
use_nested_quant: False
TrainingArguments:

num_train_epochs: 1
per_device_train_batch_size: 4
gradient_accumulation_steps: 1
learning_rate: 2e-4
weight_decay: 0.001
fp16: False
group_by_length: True
lr_scheduler_type: "cosine"

Notes
Device Compatibility: The script checks GPU compatibility for bfloat16 precision. Ensure you are running on compatible hardware (e.g., NVIDIA A100).
Sequence Packing: Sequence packing is disabled by default. You can enable it if needed for your dataset.
Logging and Saving: Checkpoints and logs are saved periodically based on the configured save_steps and logging_steps.
Outputs
The fine-tuned model and results will be stored in the ./results or ./output directory (configurable).

Future Work
Expand training epochs for better model fine-tuning.
Experiment with different datasets or larger batch sizes.
Test different LoRA and quantization configurations for optimization.
