# lumi-hf-trl-dpo

Example scripts for running [HF TRL](https://github.com/huggingface/trl) [DPO](https://huggingface.co/docs/trl/dpo_trainer) training on the [LUMI supercomputer](https://www.lumi-supercomputer.eu/) with TildeOpen64K.

Memory hacks used:

- The setup uses ZeRO stage 3. This shards parameters, gradients and optimizer states across (afaik)all processes.
- DPOTrainer's built in gradient_checkpointing.

Unfortunately attempts to perform tensor parallelism and pipeline parallelism failed.

## Running

To run the example just clone the thing on LUMI and run:

```bash
cd examples/8ki_8node/
./submit_and_tail.sh launch.sh
```

This runs HF TRL DPO on 640 mock samples, of roughly 8192 token length, over 8 LUMI GPU nodes.

To perform a serious run, you'll have to change the `.py` file to load the specific data you want to load.

## Setup I tested on:

- PyTorch 2.7.1 + ROCm 6.2.4
- Transformers 5.0.0
- TRL 0.29.0
- Flash Attention 2.7.0

All of this is available on the singularity container: /scratch/project_465002038/environment/containers/rocm624_torch271.sif

## Gotchas

- **Flash Attention requires explicit `"dtype": "bfloat16"`** in `model_init_kwargs` — otherwise you get dtype mismatch errors at runtime.
- **Stale torch extension cache** — if you get mysterious crashes between runs, try clearing `~/.cache/torch_extensions` and `~/.aiter/jit/build`. These can hold broken JIT artifacts from previous failed builds.