import torch
from ddpm import DDPM
from dit_model import DiT_Llama


# turn off autograd
torch.set_grad_enabled(False)

ddpm = DDPM(
            DiT_Llama(4, dim=1024, n_layers=12, n_heads=16, num_classes=1000),
            1000,
        ).cuda()
ddpm.load_state_dict(torch.load("./ckpt/pytorch_model.bin", map_location='cpu'))

#print(ddpm)
x = torch.randn(2, 4, 32, 32).cuda()
t = torch.randint(0, 100, (2,)).cuda()
y = torch.randint(0, 10, (2,)).cuda()

out = ddpm.eps_model(x, t, y)
print(out.shape)

outs = ddpm.sample(2, (4, 32, 32), "cuda:0") / 0.13025


from diffusers.models import AutoencoderKL
from diffusers.image_processor import VaeImageProcessor

vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae").to("cuda:0")

# example decoding
x = vae.decode(outs[1:2].cuda()).sample
img = VaeImageProcessor().postprocess(image = x.detach(), do_denormalize = [True, True])[0]
img.save("5th_image.png")