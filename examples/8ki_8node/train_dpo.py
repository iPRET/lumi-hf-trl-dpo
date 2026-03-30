from transformers import AutoTokenizer
from trl import DPOTrainer, DPOConfig
from datasets import Dataset

# ---------------------------------------------------------------------------
# Configuration — adjust these to match your setup
# ---------------------------------------------------------------------------
MODEL_PATH = "/scratch/project_465002038/checkpoints/TildeOpenHF5_0_64K447700_MagnumOpusNoControl"
OUTPUT_DIR = "/scratch/project_465002038/IP/trl_test_out"
NUM_SAMPLES = 640
MAX_LENGTH = 65536
SEQ_LENGTH = 8192  # tokens per sample

# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
# TRL requires pad_token_id to be set, but many models don't ship with one.
# Without this you get: ValueError: `pad_token_id` is missing in the `processing_class`.
# We set it by ID rather than by string to avoid registering a special token
# that could be injected through user input.
tokenizer.pad_token_id = 48  # eos token id

# ---------------------------------------------------------------------------
# Dataset — replace this with your real preference data
# ---------------------------------------------------------------------------
# Dummy data: each sample has a prompt plus chosen/rejected completions
# of SEQ_LENGTH tokens. Swap this out for your actual DPO dataset.
data = [
    {
        "prompt": f"Example prompt {i}:",
        "chosen": "1" * SEQ_LENGTH,
        "rejected": "2" * SEQ_LENGTH,
    }
    for i in range(NUM_SAMPLES)
]
train_dataset = Dataset.from_list(data)

# ---------------------------------------------------------------------------
# DeepSpeed ZeRO-3 config (inlined)
# ---------------------------------------------------------------------------
ds_config = {
    "zero_optimization": {
        "stage": 3,
        "reduce_bucket_size": 5e8,
    },
    "train_micro_batch_size_per_gpu": 1,
    "bf16": {"enabled": True},
    "num_processes": 64,
    "num_machines": 8,
}

# ---------------------------------------------------------------------------
# DPO training
# ---------------------------------------------------------------------------
training_args = DPOConfig(
    output_dir=OUTPUT_DIR,
    max_length=MAX_LENGTH,  # defaults to 1024 if not set — will silently clip your sequences
    bf16=True,
    deepspeed=ds_config,
    per_device_train_batch_size=1,
    model_init_kwargs={
        "attn_implementation": "flash_attention_2",
        "dtype": "bfloat16",
    },
    gradient_checkpointing=True,
    num_train_epochs=1,  # defaults to 3 if not set
)

trainer = DPOTrainer(
    MODEL_PATH,
    ref_model=None,
    processing_class=tokenizer,
    args=training_args,
    train_dataset=train_dataset,
)

trainer.train()
