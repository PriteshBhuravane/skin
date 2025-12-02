"""
SkinFusion-Net Model Definition for Inference
"""
import torch
import torch.nn as nn
import timm

class SkinFusionNet(nn.Module):
    def __init__(self, backbone_names, num_classes=7, hidden_dim=1024, dropout=0.3):
        super().__init__()
        self.backbone_names = backbone_names
        self.backbones = nn.ModuleList()
        dims = []
        
        for name in backbone_names:
            m = self._create_backbone(name, pretrained=False)
            self.backbones.append(m)
            dim = getattr(m, 'num_features', None)
            if dim is None:
                try:
                    dim = m.get_classifier().in_features
                except Exception:
                    raise RuntimeError(f"Cannot infer num_features for {name}")
            dims.append(dim)
        
        total_dim = sum(dims)
        
        self.fusion = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5)
        )
        self.classifier = nn.Linear(hidden_dim // 2, num_classes)
    
    def _create_backbone(self, name, pretrained):
        return timm.create_model(name, pretrained=pretrained, num_classes=0, global_pool='avg')
    
    def forward(self, x):
        feats = [m(x) for m in self.backbones]
        fused = torch.cat(feats, dim=1)
        fused = self.fusion(fused)
        return self.classifier(fused)