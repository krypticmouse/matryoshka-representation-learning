import torch
from torch import nn

from dataset import get_dataset
from MRL.layer import EmotionClassifier

def test(model: nn.Module):
    _, _, testloader = get_dataset()

    criterion = nn.CrossEntropyLoss()

    test_loss = 0
    with torch.no_grad():
        for data in testloader:
            input_ids = data["input_ids"].to("cuda")
            attention_mask = data["attention_mask"].to("cuda")
            targets = data["label"].to("cuda")

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, targets)
            test_loss += loss.item()

    return test_loss / len(testloader)

if __name__ == "__main__":
    model = EmotionClassifier(model_name="bert-base-uncased", num_classes=6, apply_mrl=True)
    model.load_state_dict(torch.load("model.pth"))
    model.to("cuda")

    test_loss = test(model)
    print(f"Test Loss: {test_loss}")