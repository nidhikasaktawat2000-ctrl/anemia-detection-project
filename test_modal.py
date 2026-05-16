import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os

model = models.efficientnet_b0(weights=None)
model.classifier = nn.Sequential(nn.Dropout(0.4), nn.Linear(1280, 1))
model.load_state_dict(torch.load('model/anemia_model.pth', map_location='cpu'))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

print('--- ANEMIA images ---')
for f in os.listdir('data/anemia')[:5]:
    img = Image.open(f'data/anemia/{f}').convert('RGB')
    t = transform(img).unsqueeze(0)
    with torch.no_grad():
        prob = torch.sigmoid(model(t)).item()
    label = 'Anemia' if prob < 0.5 else 'Non-Anemia'
    print(f'  prob={prob:.2f} -> {label}')

print('--- NON-ANEMIA images ---')
for f in os.listdir('data/non_anemia')[:5]:
    img = Image.open(f'data/non_anemia/{f}').convert('RGB')
    t = transform(img).unsqueeze(0)
    with torch.no_grad():
        prob = torch.sigmoid(model(t)).item()
    label = 'Anemia' if prob < 0.5 else 'Non-Anemia'
    print(f'  prob={prob:.2f} -> {label}')