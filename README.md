# k_euler_a_smea
A sampler based on Euler_a that applies a sinusoidal schedule for multiple passes.

采样器代码基于k_diffusion库中的euler a，因为对comfyui的插件开发了解较少，所以为了可以兼容comfyui，使用了 https://github.com/Koishi-Star/Euler-Smea-Dyn-Sampler 中的部分代码来提供对comfyui的兼容。此采样器在群友的帮助下完成。

The sampler code is based on euler a in the k_diffusion library. Because I know less about developing plugins for comfyui, I used some of the code in https://github.com/Koishi-Star/Euler-Smea-Dyn-Sampler to provide compatibility with comfyui. This project was completed with the help of a qq group members.

# 注意，此采样器会造成两倍性能损耗！
# This sampler will consume twice the performance.

### 经过测试，k_euler_a_smea采样器可以一定程度上提高高分辨率下生成图片的肢体稳定性，改善了双头，多肢等情况。
### 同时，此采样器可以使画面变得更加柔和，降低画面整体的对比度，防止过饱和。
### 不过经过部分人反馈，使用此采样器生成的图片会有模糊的感觉。不过我个人还是挺喜欢这种感觉的。
After testing, the k_euler_a_smea sampler can improve the limb stability of the generated picture at high resolutions, improving situations such as double heads and multiple limbs. At the same time, this sampler can make the picture softer, reduce the overall contrast of the picture, and prevent oversaturation.
However, after feedback from some people, the pictures generated using this sampler can feel blurry. But I personally still like this feeling.


