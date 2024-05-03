import torch

from ddpm import DDPM
from dit_model import DiT_Llama

# turn off autograd
torch.set_grad_enabled(False)
with torch.no_grad():

    ddpm = DDPM(
        DiT_Llama(4, dim=1024, n_layers=12, n_heads=16, num_classes=1000),
        1000,
    ).cuda()
    ddpm.load_state_dict(
        torch.load(
            "/mnt/chatbot30TB/simoryu/debugs/imgnet/min-max-in-dit/ckpt/model_48001/ema2.pt",
            map_location="cuda",
        ),
        strict=False,
    )

    # print(ddpm)
    x = torch.randn(2, 4, 32, 32).cuda()
    t = torch.randint(0, 100, (2,)).cuda()
    y = torch.randint(0, 10, (2,)).cuda()

    out = ddpm.eps_model(x, t, y)
    print(out.shape)

    conds = [933, 849, 94, 230, 934]
    outs = ddpm.sample((4, 32, 32), "cuda:0", conds) / 0.13025

    from diffusers.image_processor import VaeImageProcessor
    from diffusers.models import AutoencoderKL

    vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae").to("cuda:0")

    # example decoding
    for i, labidx in enumerate(conds):
        x = vae.decode(outs[i : i + 1].cuda()).sample
        img = VaeImageProcessor().postprocess(
            image=x.detach(), do_denormalize=[True, True]
        )[0]
        img.save(f"{labidx}th_image.png")
