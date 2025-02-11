# k_euler_a_smea - ComfyUI Only!!!
A sampler based on Euler_a that applies a sinusoidal schedule for multiple passes.

采样器代码基于k_diffusion库中的euler a，因为对comfyui的插件开发了解较少，所以为了可以兼容comfyui，使用了 https://github.com/Koishi-Star/Euler-Smea-Dyn-Sampler 中的部分代码来提供对comfyui的兼容。此采样器在群友的帮助下完成。

The sampler code is based on euler a in the k_diffusion library. Because I know less about developing plugins for comfyui, I used some of the code in https://github.com/Koishi-Star/Euler-Smea-Dyn-Sampler to provide compatibility with comfyui. This project was completed with the help of a qq group members.

# 注意，此采样器会造成两倍性能损耗！
# This sampler will consume twice the performance.

### k_euler_a_smea采样器可以一定程度上提高高分辨率下生成图片的肢体稳定性，可能改善双头，多肢等情况（em。。。也说不定会更糟。还需要测试）。
### 同时，此采样器可以使画面变得更加柔和，降低画面整体的对比度，防止过饱和。
### 不过经过部分人反馈，使用此采样器生成的图片会有模糊的感觉。不过我个人还是挺喜欢这种感觉的。


