from torch import nn
from transformers import AutoModel

class EmotionClassifier(nn.Module):
    def __init__(self, model_name, num_classes, m = [16, 64, 128, 256, 512, 768], apply_mrl = True):
        super(EmotionClassifier, self).__init__()

        self.model = AutoModel.from_pretrained(model_name)
        self.m = m

        if apply_mrl:
            for doll in m:
                setattr(f"classifier_{doll}", nn.Linear(doll, num_classes))
        else:
            setattr(f"classifier", nn.Linear(768, num_classes))

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        last_hidden_state = outputs.last_hidden_state
        cls_output = last_hidden_state[:, 0, :]
        
        if self.apply_mrl:
            if self.training:
                logits = [getattr(f"classifier_{doll}")(cls_output[:doll]) for doll in self.m]
            else:
                if isinstance(self.m, list):
                    dim = dim[-1]
                logits = getattr(f"classifier_{dim}")(cls_output[:dim])
        else:
            logits = self.classifier(cls_output)
        
        return logits