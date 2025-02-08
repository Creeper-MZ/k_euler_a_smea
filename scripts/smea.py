try:
    import smea_sampling
    from smea_sampling import sample_euler_a_smea,sample_euler_smea_dy_nai,sample_euler_dy, sample_euler_smea_dy, sample_euler_negative, sample_euler_dy_negative, sample_Kohaku_LoNyu_Yog
    if smea_sampling.BACKEND == "WebUI":
        from modules import scripts, sd_samplers_common, sd_samplers
        from modules.sd_samplers_kdiffusion import sampler_extra_params, KDiffusionSampler

        class SMEA(scripts.Script):
            def title(self):
                "SMEA Samplers"

            def show(self, is_img2img):
                return False

            def __init__(self):
                if not smea_sampling.INITIALIZED:
                    samplers_smea = [
                        ("sample_euler_a_smea", sample_euler_a_smea, ["sample_euler_a_smea"], {})
                    ]
                    samplers_data_smea = [
                        sd_samplers_common.SamplerData(label, lambda model, funcname=funcname: KDiffusionSampler(funcname, model), aliases, options)
                        for label, funcname, aliases, options in samplers_smea
                        if callable(funcname)
                    ]
                    sampler_extra_params["sample_euler_a_smea"] = []
                    sd_samplers.all_samplers.extend(samplers_data_smea)
                    sd_samplers.all_samplers_map = {x.name: x for x in sd_samplers.all_samplers}
                    sd_samplers.set_samplers()
                    smea_sampling.INITIALIZED = True

except ImportError as _:
    pass
