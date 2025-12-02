"""
Script to verify and fix model weights
"""
import torch
from model import SkinFusionNet

BACKBONES = [
    'convnext_base.fb_in22k_ft_in1k',
    'tf_efficientnet_b3_ns',
    'resnet50'
]

MODEL_PATH = "best_skfusion.pth"

print("🔍 Checking model...")
print(f"Model file: {MODEL_PATH}")

# Check if file exists
import os
if not os.path.exists(MODEL_PATH):
    print(f"❌ Model file not found: {MODEL_PATH}")
    print("\n📝 Please copy your trained model file 'best_skfusion.pth' to the backend folder")
    exit(1)

# Check file size
file_size = os.path.getsize(MODEL_PATH) / (1024 * 1024)  # MB
print(f"✅ File exists: {file_size:.2f} MB")

if file_size < 400:
    print("⚠️  Warning: File size seems small. Expected ~480-500 MB")

# Try loading model
print("\n🏗️  Building model architecture...")
model = SkinFusionNet(BACKBONES, num_classes=7, hidden_dim=1024, dropout=0.3)

print("📦 Loading weights...")
try:
    state_dict = torch.load(MODEL_PATH, map_location='cpu')
    
    # Check state dict
    print(f"✅ State dict loaded: {len(state_dict)} keys")
    
    # Try strict loading
    try:
        model.load_state_dict(state_dict, strict=True)
        print("✅ Model loaded successfully with strict=True!")
    except Exception as e:
        print(f"⚠️  Strict loading failed: {e}")
        print("\n🔧 Trying with strict=False...")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"⚠️  Missing keys: {len(missing)}")
            for key in missing[:5]:  # Show first 5
                print(f"   - {key}")
        if unexpected:
            print(f"⚠️  Unexpected keys: {len(unexpected)}")
            for key in unexpected[:5]:
                print(f"   - {key}")
        print("✅ Model loaded with strict=False")
    
    # Test inference
    print("\n🧪 Testing inference...")
    model.eval()
    dummy_input = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"✅ Output shape: {output.shape}")
    print(f"✅ Expected shape: torch.Size([1, 7])")
    
    if output.shape == torch.Size([1, 7]):
        print("\n🎉 Model is working correctly!")
        
        # Show sample prediction
        import torch.nn.functional as F
        probs = F.softmax(output, dim=1)[0]
        pred_class = int(probs.argmax())
        confidence = float(probs[pred_class])
        
        CLASS_NAMES = {0: 'akiec', 1: 'bcc', 2: 'bkl', 3: 'df', 4: 'mel', 5: 'nv', 6: 'vasc'}
        print(f"\n📊 Sample prediction (random input):")
        print(f"   Class: {pred_class} ({CLASS_NAMES[pred_class]})")
        print(f"   Confidence: {confidence*100:.2f}%")
        print(f"   All probs: {[f'{p*100:.2f}%' for p in probs.tolist()]}")
    else:
        print("\n❌ Output shape mismatch! Model architecture issue.")
        
except Exception as e:
    print(f"\n❌ Error loading model: {e}")
    import traceback
    traceback.print_exc()
    
    print("\n💡 Possible solutions:")
    print("1. Make sure you copied the correct model file from your training")
    print("2. Check that the model was trained with the same architecture")
    print("3. Verify BACKBONES match your training configuration")
    print("4. Try retraining the model if the architecture has changed")