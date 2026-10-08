"""The network: SegFormer-B0 trained on Cityscapes, one class per pixel.

    nvidia/segformer-b0-finetuned-cityscapes-1024-1024   (Hugging Face)

E. Xie et al., "SegFormer: Simple and Efficient Design for Semantic
Segmentation with Transformers", NeurIPS 2021. A Transformer encoder
(attention between image patches, as in L4's ViT, at four scales) and a small
decoder made of MLP layers (the model card: "a lightweight all-MLP decode
head"). The decoder gives 19 scores per pixel, one per
Cityscapes class, at a quarter of the image's size; we resize the scores to
the full image and keep the class with the highest score.

License: NVIDIA Source Code License for SegFormer, section 3.3: "non-commercially
means for research or evaluation purposes only". Course use is evaluation.
The weights (15 MB) download on first use into weights_dir, never into the repo.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

MODEL_ID = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
WEIGHTS_DIR = str(Path.home() / ".cache" / "enpm818z-weights" / "huggingface")
# The model's preprocessor_config.json: ImageNet mean and standard deviation.
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class SegFormer:

    def __init__(self, model_id: str = MODEL_ID, weights_dir: str = WEIGHTS_DIR,
                 device: str = "cuda", half: bool = True) -> None:
        import torch
        try:
            from transformers import SegformerForSemanticSegmentation
        except ImportError as exc:
            raise SystemExit(
                "l5_seg_demo needs the transformers package:\n"
                "    python3 -m pip install --user --break-system-packages transformers"
            ) from exc
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        self.torch = torch
        self.device = device
        self.half = half and device.startswith("cuda")
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            model_id, cache_dir=os.path.expanduser(weights_dir)).to(device).eval()
        self.mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(STD, device=device).view(1, 3, 1, 1)

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        """rgb: H x W x 3, uint8. Returns H x W trainIds (0 to 18), uint8.

        The image goes in at its own size, not resized to the 512 x 512 of the
        model's preprocessor: SegFormer takes any size, and the classes come
        back for every pixel of the camera.
        """
        torch = self.torch
        h, w = rgb.shape[:2]
        x = torch.from_numpy(np.ascontiguousarray(rgb)).to(self.device)
        x = x.permute(2, 0, 1)[None].float() / 255.0
        x = (x - self.mean) / self.std
        with torch.inference_mode(), torch.autocast(
                "cuda", dtype=torch.float16, enabled=self.half):
            logits = self.model(pixel_values=x).logits            # 1 x 19 x h/4 x w/4
            logits = torch.nn.functional.interpolate(
                logits.float(), size=(h, w), mode="bilinear", align_corners=False)
            return logits.argmax(1)[0].to(torch.uint8).cpu().numpy()

    def sync(self) -> None:
        """Wait for the GPU, for honest timing."""
        if self.device.startswith("cuda"):
            self.torch.cuda.synchronize()
