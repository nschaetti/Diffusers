#
#
#

from dataclasses import dataclass
from datasets import load_dataset
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import transforms
from diffusers import UNet2DModel, DDPMScheduler


@dataclass
class TrainingConfig:

    # Generated image resolution
    image_size = 128

    train_batch_size = 16

    # How many images to samples during evaluation
    eval_batch_size = 16

    num_epochs = 50

    gradient_accumulation_steps = 1

    learning_rate = 1e-4

    lr_warmup_steps = 500

    save_image_epochs = 10

    save_model_epochs = 30

    # 'no' for float32, 'fp16' for automatic
    # mixed precision
    mixed_precision = 'fp16'

    # The model namy locally and on the
    # HF Hub
    output_dir = 'ddpm-butterflies-128'

    push_to_hub = True

    # Overwrite the old model when re-running
    # the notebook
    hub_private_repo = False

    overwrite_output_dir = True

    seed = 0
# end TrainingConfig

# Training
config = TrainingConfig()

config.dataset_name = "huggan/smithsonian_butterflies_subset"
dataset = load_dataset(config.dataset_name, split="train")

# Or just load images from a local folder!
# config.dataset_name = "imagefolder"
# dataset = load_dataset(config.dataset_name, data_dir="path/to/folder")

fig, axs = plt.subplots(1, 4, figsize=(16, 4))
for i, image in enumerate(dataset[:4]["image"]):
    axs[i].imshow(image)
    axs[i].set_axis_off()
# end for
fig.show()

# Preprocessing
preprocess = transforms.Compose(
    [
        transforms.Resize((config.image_size, config.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ]
)


def transform(examples):
    images = [
        preprocess(image.convert("RGB")) for image in examples["image"]
    ]
    return {"images": images}
# end transforms


fix, axs = plt.subplots(1, 4, figsize=(16, 4))
for i, image in enumerate(dataset[:4]["images"]):
    axs[i].imshow(image.permute(1, 2, 0).numpy() / 2 + 0.5)
    axs[i].set_axis_off()
# end for


train_dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=config.train_batch_size,
    shuffle=True
)

#
# Defining the diffusion model
#

# Model
model = UNet2DModel(
    sample_size=config.image_size,
    in_channels=3,
    out_channels=3,
    layers_per_block=2,
    block_out_channels=(128, 128, 256, 256, 512, 512),
    down_block_types=(
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",
        "DownBlock2D"
    ),
    up_block_types=(
        "UpBlock2D",
        "AttnUpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D"
    )
)

sample_image = dataset[0]['images'].unsqueeze(0)
print(f"Input shape: {sample_image.shape}")
print(f"Output shape: {model(sample_image, timestep=0).sample.shape}")

#
# Defining the noise scheduler
#

# Noise scheduler
noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

# Noise
noise = torch.randn(sample_image.shape)
timesteps = torch.LongTensor([50])
noisy_image = noise_scheduler.add_noise(sample_image, noise, timesteps)


