import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
import random
import json
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import base64
import io

from models.model import create_model
from utils.data_loader import load_flower_dataset

app = Flask(__name__)
CORS(app)

device = torch.device('cpu')
print(f"Using device: {device}")

checkpoint = torch.load('checkpoints/best_model.pth', map_location=device)

num_classes = len(checkpoint['class_to_idx'])
class_to_idx = checkpoint['class_to_idx']
idx_to_class = {v: k for k, v in class_to_idx.items()}

class_to_chinese = {
    'daisy': '雏菊',
    'dandelion': '蒲公英',
    'rose': '玫瑰',
    'sunflower': '向日葵',
    'tulip': '郁金香'
}

model = create_model(
    num_classes=num_classes,
    model_type=checkpoint['config']['model_type'],
    pretrained=False
).to(device)

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

transform = transforms.Compose([
    transforms.Resize((checkpoint['config']['image_size'], checkpoint['config']['image_size'])),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_dir = 'data/flowers/test'
test_images = []

for img_name in os.listdir(test_dir):
    if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
        test_images.append(os.path.join(test_dir, img_name))

print(f"从 {test_dir} 目录加载了 {len(test_images)} 张测试图片")

used_indices = set()

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/get_random_image')
def get_random_image():
    global used_indices
    
    if len(used_indices) >= len(test_images):
        used_indices = set()
    
    available_indices = [i for i in range(len(test_images)) if i not in used_indices]
    if not available_indices:
        used_indices = set()
        available_indices = list(range(len(test_images)))
    
    idx = random.choice(available_indices)
    used_indices.add(idx)
    
    image_path = test_images[idx]
    
    with open(image_path, 'rb') as f:
        image_data = f.read()
    image_base64 = base64.b64encode(image_data).decode('utf-8')
    
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)
        predicted_idx = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_idx].item() * 100
    
    predictions = []
    for i in range(num_classes):
        predictions.append({
            'class': class_to_chinese[idx_to_class[i]],
            'probability': probabilities[0][i].item() * 100
        })
    
    predictions.sort(key=lambda x: x['probability'], reverse=True)
    
    return jsonify({
        'image': f'data:image/jpeg;base64,{image_base64}',
        'predicted_label': predicted_idx,
        'confidence': confidence,
        'predictions': predictions,
        'image_path': os.path.basename(image_path)
    })

@app.route('/reset')
def reset():
    global used_indices
    used_indices = set()
    return jsonify({'status': 'reset', 'message': '已重置测试'})

@app.route('/get_classes')
def get_classes():
    return jsonify(class_to_idx)

if __name__ == '__main__':
    print(f"\n{'='*60}")
    print("花卉识别测试系统")
    print(f"{'='*60}")
    print(f"测试集样本数: {len(test_images)}")
    print(f"类别数量: {num_classes}")
    print(f"类别映射: {class_to_idx}")
    print(f"{'='*60}\n")
    print("启动Web服务...")
    print("请在浏览器中访问: http://localhost:5000")
    print(f"{'='*60}\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False)
