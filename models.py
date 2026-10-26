import torch
from torch import nn

MODEL_CONFIGS={
 "m1":{"layers":2,"d_model":128,"heads":4,"d_ff":512},
 "m2":{"layers":3,"d_model":128,"heads":4,"d_ff":512},
 "m3":{"layers":4,"d_model":128,"heads":4,"d_ff":512},
}

class TinyTransformer(nn.Module):
    def __init__(self,vocab_size=49,n_classes=32,max_len=4,layers=2,d_model=128,heads=4,d_ff=512,dropout=0.0):
        super().__init__()
        self.token_emb=nn.Embedding(vocab_size,d_model); self.pos_emb=nn.Embedding(max_len,d_model)
        block=nn.TransformerEncoderLayer(d_model=d_model,nhead=heads,dim_feedforward=d_ff,
          dropout=dropout,activation="gelu",batch_first=True,norm_first=True)
        self.encoder=nn.TransformerEncoder(block,num_layers=layers,enable_nested_tensor=False)
        self.norm=nn.LayerNorm(d_model); self.head=nn.Linear(d_model,n_classes)
    def forward(self,tokens):
        pos=torch.arange(tokens.shape[1],device=tokens.device).unsqueeze(0)
        h=self.encoder(self.token_emb(tokens)+self.pos_emb(pos))
        return self.head(self.norm(h[:,-1,:]))
