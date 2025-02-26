from importlib import import_module
from tqdm.auto import trange
import torch
import math
import torch.nn.functional as F

sampling = None
BACKEND = None
INITIALIZED = False

if not BACKEND:
    try:
        _ = import_module("modules.sd_samplers_kdiffusion")
        sampling = import_module("k_diffusion.sampling")
        BACKEND = "WebUI"
    except ImportError as _:
        pass

if not BACKEND:
    try:
        sampling = import_module("comfy.k_diffusion.sampling")
        BACKEND = "ComfyUI"
    except ImportError as _:
        pass
class _Rescaler:
    def __init__(self, model, x, mode, **extra_args):
        self.model = model
        self.x = x
        self.mode = mode
        self.extra_args = extra_args
        if BACKEND == "WebUI":
            self.init_latent, self.mask, self.nmask = model.init_latent, model.mask, model.nmask
        if BACKEND == "ComfyUI":
            self.latent_image, self.noise = model.latent_image, model.noise
            self.denoise_mask = self.extra_args.get("denoise_mask", None)

    def __enter__(self):
        if BACKEND == "WebUI":
            if self.init_latent is not None:
                self.model.init_latent = torch.nn.functional.interpolate(input=self.init_latent, size=self.x.shape[2:4], mode=self.mode)
            if self.mask is not None:
                self.model.mask = torch.nn.functional.interpolate(input=self.mask.unsqueeze(0), size=self.x.shape[2:4], mode=self.mode).squeeze(0)
            if self.nmask is not None:
                self.model.nmask = torch.nn.functional.interpolate(input=self.nmask.unsqueeze(0), size=self.x.shape[2:4], mode=self.mode).squeeze(0)
        if BACKEND == "ComfyUI":
            if self.latent_image is not None:
                self.model.latent_image = torch.nn.functional.interpolate(input=self.latent_image, size=self.x.shape[2:4], mode=self.mode)
            if self.noise is not None:
                self.model.noise = torch.nn.functional.interpolate(input=self.latent_image, size=self.x.shape[2:4], mode=self.mode)
            if self.denoise_mask is not None:
                self.extra_args["denoise_mask"] = torch.nn.functional.interpolate(input=self.denoise_mask, size=self.x.shape[2:4], mode=self.mode)

        return self

    def __exit__(self, type, value, traceback):
        if BACKEND == "WebUI":
            del self.model.init_latent, self.model.mask, self.model.nmask
            self.model.init_latent, self.model.mask, self.model.nmask = self.init_latent, self.mask, self.nmask
        if BACKEND == "ComfyUI":
            del self.model.latent_image, self.model.noise
            self.model.latent_image, self.model.noise = self.latent_image, self.noise

def default_noise_sampler(x):
    return lambda sigma, sigma_next: torch.randn_like(x)


def get_ancestral_step(sigma_from, sigma_to, eta=1.):
    """Calculates the noise level (sigma_down) to step down to and the amount
    of noise to add (sigma_up) when doing an ancestral sampling step."""
    if not eta:
        return sigma_to, 0.
    sigma_up = min(sigma_to, eta * (sigma_to ** 2 * (sigma_from ** 2 - sigma_to ** 2) / sigma_from ** 2) ** 0.5)
    sigma_down = (sigma_to ** 2 - sigma_up ** 2) ** 0.5
    return sigma_down, sigma_up



def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f'input has {x.ndim} dims but target_dims is {target_dims}, which is less')
    expanded = x[(...,) + (None,) * dims_to_append]
    # MPS will get inf values if it tries to index into the new axes, but detaching fixes this.
    # https://github.com/pytorch/pytorch/issues/84364
    return expanded.detach().clone() if expanded.device.type == 'mps' else expanded


def to_d(x, sigma, denoised):
    """Converts a denoiser output to a Karras ODE derivative."""
    return (x - denoised) / append_dims(sigma, x.ndim).to(x.device)

def default_mid_func(i, sigma_i, sigma_down, n_steps):
    progress = i / float(n_steps - 1) if (n_steps > 1) else 0
    w = math.sin(progress * math.pi * 0.5+math.pi/6) ** 2
    return sigma_down*(1-w) + sigma_i*(w)
@torch.no_grad()
def dy_sampling_step(x, model, dt, sigma_hat, **extra_args):
    original_shape = x.shape
    batch_size, channels, m, n = original_shape[0], original_shape[1], original_shape[2] // 2, original_shape[3] // 2
    extra_row = x.shape[2] % 2 == 1
    extra_col = x.shape[3] % 2 == 1

    if extra_row:
        extra_row_content = x[:, :, -1:, :]
        x = x[:, :, :-1, :]
    if extra_col:
        extra_col_content = x[:, :, :, -1:]
        x = x[:, :, :, :-1]

    a_list = x.unfold(2, 2, 2).unfold(3, 2, 2).contiguous().view(batch_size, channels, m * n, 2, 2)
    c = a_list[:, :, :, 1, 1].view(batch_size, channels, m, n)

    with _Rescaler(model, c, 'nearest-exact', **extra_args) as rescaler:
        denoised = model(c, sigma_hat * c.new_ones([c.shape[0]]), **rescaler.extra_args)
    d = sampling.to_d(c, sigma_hat, denoised)
    c = c + d * dt

    d_list = c.view(batch_size, channels, m * n, 1, 1)
    a_list[:, :, :, 1, 1] = d_list[:, :, :, 0, 0]
    x = a_list.view(batch_size, channels, m, n, 2, 2).permute(0, 1, 2, 4, 3, 5).reshape(batch_size, channels, 2 * m, 2 * n)

    if extra_row or extra_col:
        x_expanded = torch.zeros(original_shape, dtype=x.dtype, device=x.device)
        x_expanded[:, :, :2 * m, :2 * n] = x
        if extra_row:
            x_expanded[:, :, -1:, :2 * n + 1] = extra_row_content
        if extra_col:
            x_expanded[:, :, :2 * m, -1:] = extra_col_content
        if extra_row and extra_col:
            x_expanded[:, :, -1:, -1:] = extra_col_content[:, :, -1:, :]
        x = x_expanded

    return x,denoised
@torch.no_grad()
def sample_euler_a_smea(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=False,
    eta=1.,
    s_noise=1.,
    noise_sampler=None,
    mid_func=None
):
    if extra_args is None:
        extra_args = {}
    if noise_sampler is None:
        noise_sampler = default_noise_sampler(x)

    s_in = x.new_ones([x.shape[0]],device=x.device)
    n_steps = len(sigmas) - 1

    print(extra_args,eta,s_noise)

    if mid_func is None:
        mid_func = default_mid_func

    for i in trange(n_steps, disable=disable):
        sigma_i = sigmas[i]
        sigma_next = sigmas[i+1]

        sigma_down, sigma_up = get_ancestral_step(sigma_i, sigma_next, eta=eta)

        # 1) 计算中间sigma
        sigma_mid = torch.tensor(mid_func(i, sigma_i, sigma_down, n_steps), dtype=x.dtype, device=x.device)
        if sigma_down < sigma_i:
            sigma_mid = torch.max(torch.min(sigma_mid, sigma_i), sigma_down).to(x.device)
        else:
            sigma_mid = torch.min(torch.max(sigma_mid, sigma_i), sigma_down).to(x.device)

        # 子步 A: from sigma_i -> sigma_mid
        factor = 1
        if i<(n_steps-n_steps/10):
            factor = math.sin((i/(n_steps-n_steps/10))*(math.pi/2)) * 0.6 + 0.5
            print(factor)
        x_s = F.interpolate(x, scale_factor=factor, mode='nearest').to(x.device)
        denoised_a = model(x_s, sigma_i*s_in, **extra_args).to(x.device)  # U-Net预测
        denoised_a = F.interpolate(denoised_a, size=x.shape[2:], mode='nearest').to(x.device)
        d_a = to_d(x, sigma_i, denoised_a).to(x.device)
        # 计算子步更新:
        dt_a = sigma_mid - sigma_i
        x = x + d_a * dt_a
        #gamma = torch.min(sigma_mid, eta * (sigma_mid ** 2 * (sigma_i ** 2 - sigma_mid ** 2) / sigma_i ** 2) ** 0.5)
        #x = x + noise_sampler(sigma_i,sigma_mid)*s_noise*gamma*((1.1-factor)/2)
        # 子步 B: from sigma_mid -> sigma_down
        dt_b = sigma_down - sigma_mid
        if i<0:
            x,denoised_b=dy_sampling_step(x,model,dt_b,sigma_mid,**extra_args)
        else:
            denoised_b = model(x, sigma_mid*s_in, **extra_args).to(x.device)
            d_b = to_d(x, sigma_mid, denoised_b).to(x.device)
            x = x + d_b * dt_b
        print(sigma_i,sigma_mid,sigma_next)



        #加 euler a 的祖先噪声
        if sigma_next > 0:
            x = x + noise_sampler(sigma_i, sigma_next)*s_noise*sigma_up
        if callback is not None:
            callback({
                'x': x,
                'i': i,
                'sigma': sigmas[i],
                'sigma_hat': sigmas[i],
                'denoised': denoised_b
            })

    return x