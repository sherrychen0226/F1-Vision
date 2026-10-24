import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
import gradio as gr

TEAM_MODEL_PATH = "checkpoints/resnet18_best.pth"
TEAM_CLASSES = ["Ferrari", "McLaren", "Mercedes", "RedBull"]

DRIVER_MODELS = {
    "Ferrari": {
        "path": "checkpoints/ferrari_driver_best.pth",
        "classes": ["Hamilton", "Leclerc"]
    },
    "McLaren": {
        "path": "checkpoints/mclaren_driver_best.pth",
        "classes": ["Norris", "Piastri"]
    },
    "Mercedes": {
        "path": "checkpoints/mercedes_driver_best.pth",
        "classes": ["Antonelli", "Russell"]
    },
    "RedBull": None
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights = ResNet18_Weights.DEFAULT

team_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    weights.transforms()
])

driver_transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def load_resnet_model(model_path, num_classes):
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model


team_model = load_resnet_model(TEAM_MODEL_PATH, len(TEAM_CLASSES))

driver_models = {}

for team, info in DRIVER_MODELS.items():
    if info is not None:
        driver_models[team] = load_resnet_model(
            info["path"],
            len(info["classes"])
        )


def predict(image):
    image = image.convert("RGB")

    # ==========================
    # Stage 1: Team Prediction
    # ==========================
    team_img = team_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        team_outputs = team_model(team_img)
        team_probs = torch.softmax(team_outputs, dim=1)[0]

    team_idx = torch.argmax(team_probs).item()
    predicted_team = TEAM_CLASSES[team_idx]
    team_confidence = team_probs[team_idx].item() * 100

    team_prob_dict = {
        TEAM_CLASSES[i]: float(team_probs[i])
        for i in range(len(TEAM_CLASSES))
    }

    team_result = f"""
# 🏎️ Team Prediction

## Prediction: **{predicted_team}**

## Confidence: **{team_confidence:.2f}%**
"""

    # ==========================
    # Stage 2: Driver Prediction
    # ==========================
    if DRIVER_MODELS[predicted_team] is None:
        driver_result = f"""
# 👤 Driver Prediction

## Status: **Coming Soon**

Driver prediction for **{predicted_team}** is not available yet.
"""
        driver_prob_dict = {"Coming Soon": 1.0}
        return team_result, team_prob_dict, driver_result, driver_prob_dict

    driver_info = DRIVER_MODELS[predicted_team]
    driver_classes = driver_info["classes"]
    driver_model = driver_models[predicted_team]

    driver_img = driver_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        driver_outputs = driver_model(driver_img)
        driver_probs = torch.softmax(driver_outputs, dim=1)[0]

    driver_idx = torch.argmax(driver_probs).item()
    predicted_driver = driver_classes[driver_idx]
    driver_confidence = driver_probs[driver_idx].item() * 100

    driver_prob_dict = {
        driver_classes[i]: float(driver_probs[i])
        for i in range(len(driver_classes))
    }

    driver_result = f"""
# 👤 Driver Prediction

## Prediction: **{predicted_driver}**

## Confidence: **{driver_confidence:.2f}%**
"""

    return team_result, team_prob_dict, driver_result, driver_prob_dict


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload an F1 car image"),
    outputs=[
        gr.Markdown(label="Team Prediction Result"),
        gr.Label(num_top_classes=4, label="Team Probabilities"),
        gr.Markdown(label="Driver Prediction Result"),
        gr.Label(num_top_classes=2, label="Driver Probabilities")
    ],
    title="🏎️ F1 Vision V2",
    description="Upload an F1 car image. The system first predicts the team, then predicts the driver if the driver model is available."
)

demo.launch()