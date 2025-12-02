"""
Test script to verify skin detector is working.
Run this in backend folder: python3 test_skin_detector.py
"""

import os
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import timm


class SkinDetector(nn.Module):
    def __init__(self, backbone="efficientnet_b0", dropout=0.3):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=False, num_classes=0)

        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            features = self.backbone(dummy)
            feature_dim = features.shape[1]

        print(f"Backbone feature dimension: {feature_dim}")

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features).squeeze()


def test_skin_detector():
    print("=" * 80)
    print("SKIN DETECTOR TEST")
    print("=" * 80)

    SKIN_DETECTOR_PATH = "skin_detector.pth"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n1. Checking if model file exists...")
    if not os.path.exists(SKIN_DETECTOR_PATH):
        print(f"❌ ERROR: {SKIN_DETECTOR_PATH} not found!")
        print(f"   Current directory: {os.getcwd()}")
        print(f"   Files in directory: {os.listdir('.')}")
        return False
    else:
        print(f"✅ Found {SKIN_DETECTOR_PATH}")
        file_size = os.path.getsize(SKIN_DETECTOR_PATH) / (1024 * 1024)
        print(f"   File size: {file_size:.2f} MB")

    print("\n2. Loading model...")
    try:
        skin_detector = SkinDetector(backbone="efficientnet_b0", dropout=0.3)

        # IMPORTANT FIX: weights_only=False for PyTorch 2.6+
        checkpoint = torch.load(
            SKIN_DETECTOR_PATH, map_location=DEVICE, weights_only=False
        )

        print(f"   Checkpoint type: {type(checkpoint)}")
        if isinstance(checkpoint, dict):
            print(f"   Checkpoint keys: {checkpoint.keys()}")
            if "model_state_dict" in checkpoint:
                skin_detector.load_state_dict(checkpoint["model_state_dict"])
                print("   ✅ Loaded from checkpoint['model_state_dict']")
            else:
                skin_detector.load_state_dict(checkpoint)
                print("   ✅ Loaded directly from checkpoint dict")
        else:
            skin_detector.load_state_dict(checkpoint)
            print("   ✅ Loaded directly from checkpoint")

        skin_detector.to(DEVICE)
        skin_detector.eval()
        print(f"✅ Model loaded successfully on {DEVICE}")

    except Exception as e:
        print(f"❌ ERROR loading model: {e}")
        import traceback

        traceback.print_exc()
        return False

    print("\n3. Testing with dummy input...")
    try:
        dummy_input = torch.randn(1, 3, 224, 224).to(DEVICE)
        with torch.no_grad():
            output = skin_detector(dummy_input)
        print("✅ Model inference works!")
        print(f"   Output shape: {output.shape}")
        print(f"   Output value: {float(output):.4f}")
        print(f"   Output type: {type(float(output))}")
    except Exception as e:
        print(f"❌ ERROR during inference: {e}")
        import traceback

        traceback.print_exc()
        return False

    print("\n" + "=" * 80)
    print("✅ SKIN DETECTOR TEST PASSED!")
    print("=" * 80)
    return True


if __name__ == "__main__":
    test_skin_detector()
