import torch
import argparse

from diffusers import FluxPipeline


# Adjust these paths to your own
TOKEN = "<hf-token>"
FLUX_PATH = "black-forest-labs/FLUX.1-dev"

if __name__ == "__main__":
    # Parse cmd args
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt1", type=str, required=True)
    parser.add_argument("--prompt2", type=str, required=True)
    parser.add_argument("--min_t", type=float, default=0.0)
    parser.add_argument("--max_t", type=float, default=1.0)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--kernel_size", type=int, default=25)
    cmd_args = parser.parse_args()

    if cmd_args.kernel_size % 2 == 0:
        raise ValueError("Kernel size must be odd")

    # Load the Flux pipeline
    image_generation_pipeline = FluxPipeline.from_pretrained(FLUX_PATH, torch_dtype=torch.bfloat16).to("cuda")
    original_forward = image_generation_pipeline.transformer.forward

    avg_pool = torch.nn.AvgPool1d(kernel_size=cmd_args.kernel_size, stride=1, padding=cmd_args.kernel_size // 2)

    def cutoff_val(t, min_t, max_t):
        if t > max_t:
            return 1.0
        elif t < min_t:
            return 0.0
        else:
            return ((t - min_t) / (max_t - min_t)) ** 10.0

    def noise_unfreeze(noise, t, min_t, max_t):
        noise_std = noise.std(dim=0, keepdim=True)
        noise_std = avg_pool(noise_std)
        tau = cutoff_val(t, min_t, max_t)
        if tau == 1.0:
            noise_mean = noise.mean(dim=0).unsqueeze(0).expand_as(noise)
        else:
            noise_mean = noise[0].unsqueeze(0).expand_as(noise)

        cutoff = torch.quantile(noise_std.detach().cpu().float(), tau)
        mask = noise_std < cutoff
        noise_modified = torch.where(mask, noise_mean, noise)
        return noise_modified

    def equalizer_wrapper(*args, **kwargs):
        timestep = kwargs.get("timestep", None)
        timestep_mean = timestep.mean().item()
        output, = original_forward(**kwargs)
        return noise_unfreeze(output, timestep_mean, cmd_args.min_t, cmd_args.max_t),
    
    # Patch the transformer to use the equalizer wrapper
    image_generation_pipeline.transformer.forward = equalizer_wrapper.__get__(image_generation_pipeline.transformer, image_generation_pipeline.transformer.__class__)

    # Prepare latents
    latents0, _ = image_generation_pipeline.prepare_latents(
        2,
        image_generation_pipeline.transformer.config.in_channels // 4,
        512,
        512,
        torch.float16,
        device="cuda",
        latents=None,
        generator=None
    )

    latents0 = latents0[0].unsqueeze(0).expand_as(latents0)
    latents = latents0

    try:
        imgs = image_generation_pipeline(
            [cmd_args.prompt1, cmd_args.prompt2],
            num_inference_steps=cmd_args.num_inference_steps,
            width=512,
            height=512,
            num_images_per_prompt=1,
            latents=latents
        )
    except Exception as e:
        raise e
    finally:
        image_generation_pipeline.transformer.forward = original_forward.__get__(image_generation_pipeline.transformer, image_generation_pipeline.transformer.__class__)
    
    img1, img2 = imgs.images
    img1.save("img1.png")
    img2.save("img2.png")
