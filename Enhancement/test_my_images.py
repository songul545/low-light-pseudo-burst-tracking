import os
import argparse
from glob import glob

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from natsort import natsorted
from skimage import img_as_ubyte
from tqdm import tqdm

import utils
from basicsr.models import create_model
from basicsr.utils.options import parse


def load_model(opt_path, weights_path, device):
    opt = parse(opt_path, is_train=False)
    opt["dist"] = False

    model = create_model(opt).net_g

    checkpoint = torch.load(weights_path, map_location=device)

    try:
        model.load_state_dict(checkpoint["params"])
    except Exception:
        new_checkpoint = {}
        for k in checkpoint["params"]:
            new_checkpoint["module." + k] = checkpoint["params"][k]
        model.load_state_dict(new_checkpoint)

    model = model.to(device)
    model.eval()
    return model


def enhance_image(model, img_path, output_path, device):
    factor = 4

    img = np.float32(utils.load_img(img_path)) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1)
    input_ = img.unsqueeze(0).to(device)

    b, c, h, w = input_.shape

    H = ((h + factor) // factor) * factor
    W = ((w + factor) // factor) * factor

    padh = H - h if h % factor != 0 else 0
    padw = W - w if w % factor != 0 else 0

    input_ = F.pad(input_, (0, padw, 0, padh), "reflect")

    with torch.inference_mode():
        restored = model(input_)

    restored = restored[:, :, :h, :w]
    restored = torch.clamp(restored, 0, 1)
    restored = restored.cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    utils.save_img(output_path, img_as_ubyte(restored))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Input image folder")
    parser.add_argument("--output", type=str, required=True, help="Output folder")
    parser.add_argument("--opt", type=str, default="Options/RetinexFormer_SMID.yml")
    parser.add_argument("--weights", type=str, default="pretrained_weights/SMID.pth")
    parser.add_argument("--cpu", action="store_true")

    args = parser.parse_args()

    device = torch.device("cpu")
    print("Using device:", device)

    model = load_model(args.opt, args.weights, device)

    image_paths = natsorted(
        glob(os.path.join(args.input, "*.png"))
        + glob(os.path.join(args.input, "*.jpg"))
        + glob(os.path.join(args.input, "*.jpeg"))
    )

    print("Found images:", len(image_paths))

    for img_path in tqdm(image_paths):
        filename = os.path.splitext(os.path.basename(img_path))[0] + ".png"
        output_path = os.path.join(args.output, filename)
        enhance_image(model, img_path, output_path, device)

    print("Done. Results saved to:", args.output)


if __name__ == "__main__":
    main()