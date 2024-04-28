# Min-Max-Imagenet DiT

In a similar spirit to the Keller Jordan's [Fastest CIFAR-10 training](https://github.com/KellerJordan/cifar10-airbench), I want to be the fastest diffusion trainer in the east. I'll keep the progress here. Currently very much WIP.

Featuring:

* DeepSpeed training of [Diffusion Transformer](https://arxiv.org/abs/2212.09748). Supports Zero-1,2,3.
* CPU-offloaded, skipped EMA trick for [Karras' Post-hoc EMA analysis](https://arxiv.org/abs/2312.02696v1), where you EMA once in every `N` steps instead. You have to adjust `beta_1` and `beta_2` so they are properly accounting for the fact you skipped last `N-1` steps. Of course, saving codes are there.
* Featuring Streaming Dataset, specially my quantized [imagenet.int8](https://github.com/cloneofsimo/imagenet.int8) for insanely lightweight imagenet training.

## Dataset

Since this dataset is so small, you don't need to setup massive remote data setup stuff, just point to the `local_dir`, set `remote_dir` to `None`. 

## Running

For single-node setup, just

```bash
run.sh
```

## Whats the goal here?

My goal is to get FID score of 30 under 20 hours of training. I'll keep updating this README as I make progress.


