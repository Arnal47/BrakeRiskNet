import torch.nn as nn

class MLPBaseline(nn.Module):
    def __init__(self, input_dim):
        super().__init__(); self.backbone=nn.Sequential(nn.Linear(input_dim,64),nn.ReLU(),nn.Dropout(.15),nn.Linear(64,32),nn.ReLU()); self.classifier=nn.Linear(32,3); self.regressor=nn.Linear(32,1)
    def forward(self,x): h=self.backbone(x); return self.classifier(h),self.regressor(h).squeeze(1)

class GRUBaseline(nn.Module):
    def __init__(self,input_dim):
        super().__init__(); self.gru=nn.GRU(input_dim,64,batch_first=True); self.shared=nn.Sequential(nn.Linear(64,32),nn.ReLU(),nn.Dropout(.15)); self.classifier=nn.Linear(32,3); self.regressor=nn.Linear(32,1)
    def forward(self,x): _,h=self.gru(x); h=self.shared(h[-1]); return self.classifier(h),self.regressor(h).squeeze(1)
