from torch import nn
from transformers import AutoModel

from MRL.layer import MRLLayer

class EmotionClassifier(nn.Module):
    def __init__(self, model_name, num_classes, m = [16, 64, 128, 256, 512, 768], apply_mrl = True):
        super(EmotionClassifier, self).__init__()

        self.model = AutoModel.from_pretrained(model_name)
        self.m = m

        if apply_mrl:
            self.classifier = MRLLayer(num_classes, m)
        else:
            self.classifier = nn.Linear(768, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        last_hidden_state = outputs.last_hidden_state
        cls_output = last_hidden_state[:, 0, :]
        
        logits = self.classifier(cls_output)
        return logits