# In backend folder, create test_model.py
import torch
from model import SkinFusionNet

BACKBONES = [
    'convnext_base.fb_in22k_ft_in1k',
    'tf_efficientnet_b3_ns',
    'resnet50'
]

model = SkinFusionNet(BACKBONES, num_classes=7)
state_dict = torch.load('best_skfusion.pth', map_location='cpu')

try:
    model.load_state_dict(state_dict, strict=True)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Model loading failed: {e}")
    print("\nTrying with strict=False...")
    model.load_state_dict(state_dict, strict=False)
    print("⚠️ Model loaded with strict=False (some weights missing/extra)")