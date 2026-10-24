import sys
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights

MODEL_PATH = "checkpoints/resnet18_best.pth"
CLASSES = ["Ferrari", "McLaren", "Mercedes", "RedBull"]

weights = ResNet18_Weights.DEFAULT

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    weights.transforms()
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(weights=weights)
model.fc = nn.Linear(model.fc.in_features, len(CLASSES))

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

image_path = sys.argv[1]

img = Image.open(image_path).convert("RGB")
img = transform(img)
img = img.unsqueeze(0).to(device)

with torch.no_grad():
    outputs = model(img)
    probabilities = torch.softmax(outputs, dim=1)[0]

predicted_idx = torch.argmax(probabilities).item()
predicted_class = CLASSES[predicted_idx]
confidence = probabilities[predicted_idx].item()

print("\n🏎️ F1 Team Prediction")
print("====================")
print(f"Image: {image_path}")
print(f"Prediction: {predicted_class}")
print(f"Confidence: {confidence * 100:.2f}%")

print("\nClass Probabilities")
print("--------------------")

sorted_probs = sorted(
    zip(CLASSES, probabilities.tolist()),
    key=lambda x: x[1],
    reverse=True
)

for class_name, prob in sorted_probs:
    print(f"{class_name:8s}: {prob * 100:6.2f}%")