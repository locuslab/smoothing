# visualize noisy images
import argparse
from datasets import get_dataset, DATASETS
import torch
from torchvision.transforms import ToPILImage

parser = argparse.ArgumentParser(description='visualize noisy images')
parser.add_argument("dataset", type=str, choices=DATASETS)
parser.add_argument("outdir", type=str, help="output directory")
parser.add_argument("idx", type=int)
parser.add_argument("noise_sds", nargs='+', type=float)
parser.add_argument("--split", choices=["train", "test"], default="test")
args = parser.parse_args()

toPilImage = ToPILImage()
dataset = get_dataset(args.dataset, args.split)
image, _ = dataset[args.idx]
noise = torch.randn_like(image)
for noise_sd in args.noise_sds:
    noisy_image = torch.clamp(image + noise * noise_sd, min=0, max=1)
    pil = toPilImage(noisy_image)
    pil.save("{}/{}_{}.png".format(args.outdir, args.idx, int(noise_sd * 100)))
