# lumi-hf-trl-dpo

Scripts and configuration for running [HF TRL](https://github.com/huggingface/trl) DPO (Direct Preference Optimization) training on the [LUMI supercomputer](https://www.lumi-supercomputer.eu/).

## Examples

- **[8ki_8node](examples/8ki_8node/)** — 8-node (64 GPU) DPO training with 8192-token samples using DeepSpeed ZeRO-3

## Software stack

- PyTorch 2.7.1 + ROCm 6.2.4
- Transformers 5.0.0
- TRL 0.29.0
- Flash Attention 2.7.0
