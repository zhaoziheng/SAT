import torch.nn as nn

from .text_tower import Text_Tower

class Knowledge_Encoder(nn.Module):
    def __init__(self, biolord_checkpoint='FremyCompany/BioLORD-2023-C'):
        super().__init__()
        # LP
        self.text_tower = Text_Tower(biolord_checkpoint)
        self.projection_layer = nn.Sequential(
            nn.Linear(768, 768),
            nn.GELU(),
            nn.Linear(768, 768)
        )
        self.modality_embed = nn.Embedding(4, 768)
    
    def forward(self, text, modality):
        text_feature = self.text_tower(text)
        proj_text_feature = self.projection_layer(text_feature)
        
        modality_feature = self.modality_embed(modality)
        
        text_feature = text_feature + modality_feature
        proj_text_feature = proj_text_feature + modality_feature
        
        # return text_feature, proj_text_feature
        return proj_text_feature