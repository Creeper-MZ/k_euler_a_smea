from importlib import import_module
from tqdm.auto import trange
import torch
import math
import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur
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
                self.model.noise = torch.nn.functional.interpolate(input=self.noise, size=self.x.shape[2:4], mode=self.mode)
            if self.denoise_mask is not None:
                self.extra_args["denoise_mask"] = torch.nn.functional.interpolate(input=self.denoise_mask, size=self.x.shape[2:4], mode=self.mode)

        return self

    def __exit__(self, type, value, traceback):
        if BACKEND == "WebUI":
            del self.model.init_latent, self.model.mask, self.model.nmask
            self.model.init_latent, self.model.mask, self.model.nmask = self.init_latent, self.mask, self.nmask
        if BACKEND == "ComfyUI":
            del self.model.latent_image, self.model.noise
            self.model.latent_image, self.model.noise,self.extra_args["denoise_mask"] = self.latent_image, self.noise, self.denoise_mask

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
    eps = 1e-8
    return (x - denoised) / append_dims(torch.clamp(sigma, min=eps), x.ndim).to(x.device)

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
def sample_euler_a_smea(model, x, sigmas, extra_args=None, callback=None,
                        disable=False, eta=1., s_noise=1., noise_sampler=None):
    """
    SMEA采样器的正确实现
    """
    if extra_args is None:
        extra_args = {}
    if noise_sampler is None:
        noise_sampler = default_noise_sampler(x)

    n_steps = len(sigmas) - 1

    # 判断是否需要启用SMEA（高分辨率时自动启用）
    enable_smea = x.shape[2] >= 128 or x.shape[3] >= 128  # 对应实际分辨率≥1024

    for i in trange(n_steps, disable=disable):
        sigma_i = sigmas[i]
        sigma_next = sigmas[i + 1]
        apply_rate=0.6
        if enable_smea and i < n_steps * apply_rate and i%2==0 :  # 在前50%的步骤启用SMEA
            # SMEA多通道处理
            x,denoised = smea_step(x, model, sigma_i, sigma_next, eta, extra_args,
                          noise_sampler, s_noise, i, n_steps*apply_rate)
        else:
            # 标准Euler Ancestral步骤
            x,denoised = euler_ancestral_step(x, model, sigma_i, sigma_next, eta,
                                     extra_args, noise_sampler, s_noise)

        if callback is not None:
            callback({
                'x': x,
                'i': i,
                'sigma': sigmas[i],
                'sigma_hat': sigmas[i],
                'denoised': denoised  # 添加这一行
            })

    return x

def create_detail_channel(x):
    """通过增强高频分量来创建细节通道"""
    # 使用高斯模糊获取低频分量
    low_freq = gaussian_blur(x, kernel_size=[3,3])
    # 计算高频细节
    high_freq = x - low_freq
    # 增强高频细节
    detail_enhanced = x + 0.5 * high_freq
    return detail_enhanced
def smea_step(x, model, sigma_i, sigma_next, eta, extra_args,
              noise_sampler, s_noise, step, total_steps):
    """
    SMEA的核心多通道处理步骤
    """
    # 计算进度和正弦权重
    progress = step / float(total_steps)
    sine_weight = math.sin(progress * math.pi * 0.5) ** 2
    # 多通道评估
    n_passes = 3
    denoised_results = []
    print(sigma_i)
    print(x.shape[2:])
    for pass_idx in range(n_passes):
        # 为每个通道准备不同的输入
        if pass_idx == 0:
            # 主通道：原始分辨率
            x_input = x
            scale_factor = 1.0
        elif pass_idx == 1:
            # 全局通道：降低分辨率以捕获全局特征
            scale_factor = 0.6
            x_input = F.interpolate(x, size=[96,96],
                                    mode='nearest')

        else:
            # 细节通道
            scale_factor = 1
            x_input = create_detail_channel(x)
        if pass_idx == 1:
            print(x_input.std())
            print(sigma_i)
            print(scale_factor)
            with _Rescaler(model, x_input, 'nearest', **extra_args) as rescaler:
                denoised = model(x_input,
                                 (x_input.std()*(1-sine_weight)+sigma_i*sine_weight) * x_input.new_ones([x_input.shape[0]]),
                             **rescaler.extra_args)
        else:
            # UNet评估
            denoised = model(x_input, sigma_i * x_input.new_ones([x_input.shape[0]]),
                            **extra_args)

        # 如果缩放了，恢复到原始尺寸
        if scale_factor != 1.0:
            denoised = F.interpolate(denoised, size=x.shape[2:],
                                     mode='nearest')

        denoised_results.append(denoised)

    # 使用正弦权重组合多个结果
    if n_passes == 3:
        # 组合权重可以根据进度动态调整
        w1 =  sine_weight  # 主通道权重
        w2 = 1- sine_weight # 全局通道权重
        w3 = 0.15-0.1*sine_weight  # 细节通道权重
        print(sine_weight)
        # 归一化权重
        w_sum = w1 + w2 + w3
        w1, w2, w3 = w1 / w_sum, w2 / w_sum, w3 / w_sum
        print(w1,w2,w3)
        denoised_combined = (w1 * denoised_results[0] +
                             w2 * denoised_results[1] +
                             w3 * denoised_results[2])
    else:
        # 简单平均
        denoised_combined = sum(denoised_results) / len(denoised_results)

    # 执行更新步骤
    sigma_down, sigma_up = get_ancestral_step(sigma_i, sigma_next, eta=eta)
    d = to_d(x, sigma_i, denoised_combined)
    dt = sigma_down - sigma_i
    x = x + d * dt

    # 添加噪声
    if sigma_next > 0:
        x = x + noise_sampler(sigma_i, sigma_next) * s_noise * sigma_up

    return x, denoised_combined


def euler_ancestral_step(x, model, sigma_i, sigma_next, eta, extra_args,
                         noise_sampler, s_noise):
    """
    标准的Euler Ancestral步骤
    """
    s_in = x.new_ones([x.shape[0]])
    denoised = model(x, sigma_i * s_in, **extra_args)

    sigma_down, sigma_up = get_ancestral_step(sigma_i, sigma_next, eta=eta)
    d = to_d(x, sigma_i, denoised)
    dt = sigma_down - sigma_i
    x = x + d * dt

    if sigma_next > 0:
        x = x + noise_sampler(sigma_i, sigma_next) * s_noise * sigma_up

    return x, denoised