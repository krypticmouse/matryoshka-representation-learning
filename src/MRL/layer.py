from torch import nn

class MRLLayer(nn.Module):
    def __init__(self, num_classes, m = [16, 64, 128, 256, 512, 768]):
        super(MRLLayer, self).__init__()

        self.m = m

        for doll in m:
            setattr(self, f"mrl_classifier_{doll}", nn.Linear(doll, num_classes))

    def forward(self, x):
        if isinstance(self.m, list):
            logits = [getattr(self, f"mrl_classifier_{doll}")(x[:, :doll]) for doll in self.m]
        elif isinstance(self.m, int):
            logits = getattr(self, f"mrl_classifier_{self.m}")(x[:, :self.m])
        else:
            raise ValueError("m should be either a list or an integer")
        
        return logits