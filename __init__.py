from .smea_sampling import sample_euler_a_smea

if smea_sampling.BACKEND == "ComfyUI":
    if not smea_sampling.INITIALIZED:
        from comfy.k_diffusion import sampling as k_diffusion_sampling
        from comfy.samplers import SAMPLER_NAMES

        setattr(k_diffusion_sampling, "sample_k_euler_a_smea", sample_euler_a_smea)
        SAMPLER_NAMES.append("k_euler_a_smea")
        smea_sampling.INITIALIZED = True

NODE_CLASS_MAPPINGS = {}
