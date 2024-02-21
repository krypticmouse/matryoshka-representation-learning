from torch import nn

class MRLLayer(nn.Module):
    def __init__(self, num_classes, m = [16, 64, 128, 256, 512, 768]):
        super(MRLLayer, self).__init__()

        self.m = m

        for doll in m:
            setattr(self, f"mrl_classifier_{doll}", nn.Linear(doll, num_classes))

    def forward(self, x):
        if self.training:
            logits = [getattr(self, f"mrl_classifier_{doll}")(x[:, :doll]) for doll in self.m]
        else:
            if isinstance(self.m, list):
                dim = dim[-1]
            logits = getattr(self, f"mrl_classifier_{dim}")(x[:, :dim])
        
        return logits