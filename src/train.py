from MRL.loss import MRLLoss
from dataset import get_dataset
from engine import EmotionClassifier

import torch
from torch import nn
from transformers import AdamW, get_linear_schedule_with_warmup

def train(num_epochs: int = 10):
    trainloader, valloader, _ = get_dataset()

    model = EmotionClassifier(model_name="bert-base-uncased", num_classes=6, apply_mrl=True)
    model.to("cuda")

    criterion = MRLLoss(cm=torch.tensor([1, 1, 1, 1, 1, 1]).to("cuda"))
    optimizer = AdamW(model.parameters(), lr=1e-5)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(trainloader) * 10)

    for epoch in range(num_epochs):
        model.train()
        for data in trainloader:
            input_ids = data["input_ids"].to("cuda")
            attention_mask = data["attention_mask"].to("cuda")
            targets = data["label"].to("cuda")

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()
            scheduler.step()

        model.eval()
        with torch.no_grad():
            for data in valloader:
                input_ids = data["input_ids"].to("cuda")
                attention_mask = data["attention_mask"].to("cuda")
                targets = data["label"].to("cuda")

                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, targets)

        print(f"Epoch: {epoch}, Loss: {loss.item()}")