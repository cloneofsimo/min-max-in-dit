# Min-Max-Imagenet DiT

In a similar spirit to the Keller Jordan's [Fastest CIFAR-10 training](https://github.com/KellerJordan/cifar10-airbench), I want to be the fastest diffusion trainer in the east. I'll keep the progress here. Currently very much WIP.

## Dataset

I am currently using `imagenet.int8`, one that I made lol [check here](https://huggingface.co/datasets/cloneofsimo/imagenet.int8). Since this dataset is so small, you don't need to setup massive remote data setup stuff, just point to the `local_dir`, set `remote_dir` to `None`. 

## Running

For single-node setup, just

```bash
run.sh
```





