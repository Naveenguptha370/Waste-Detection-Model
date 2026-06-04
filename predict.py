import torch
import torch.nn as nn

from PIL import Image

from torchvision import transforms
from torchvision.models import efficientnet_v2_s

classes = [
    "dry",
    "ewaste",
    "medical",
    "sanitary",
    "wet"
]

model = efficientnet_v2_s(weights=None)

num_features = model.classifier[1].in_features

model.classifier[1] = nn.Linear(
    num_features,
    len(classes)
)

model.load_state_dict(
    torch.load(
        "waste_model.pth",
        map_location="cpu"
    )
)

model.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

image = Image.open(
    "test.jpg"
).convert("RGB")

img = transform(image)

img = img.unsqueeze(0)

with torch.no_grad():

    outputs = model(img)

    probs = torch.softmax(
        outputs,
        dim=1
    )

    confidence, pred = torch.max(
        probs,
        1
    )

print(
    f"Prediction : {classes[pred.item()]}"
)

print(
    f"Confidence : {confidence.item()*100:.2f}%"
)