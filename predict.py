import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import argparse
import os
import json

from models.model import create_model


class FlowerPredictor:
    def __init__(self, checkpoint_path, device=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.class_to_idx = checkpoint['class_to_idx']
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        self.config = checkpoint['config']

        self.model = create_model(
            num_classes=len(self.class_to_idx),
            model_type=self.config['model_type'],
            pretrained=False
        ).to(self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((self.config['image_size'], self.config['image_size'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        print(f"Model loaded successfully from {checkpoint_path}")
        print(f"Classes: {self.idx_to_class}")

    def predict_single(self, image_path, top_k=5):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert('RGB')
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
            top_probs, top_indices = torch.topk(probabilities, min(top_k, len(self.idx_to_class)))

        results = []
        for prob, idx in zip(top_probs[0], top_indices[0]):
            class_name = self.idx_to_class[idx.item()]
            confidence = prob.item() * 100
            results.append({
                'class': class_name,
                'confidence': confidence
            })

        return results, image

    def predict_batch(self, image_dir, top_k=5):
        if not os.path.exists(image_dir):
            raise FileNotFoundError(f"Directory not found: {image_dir}")

        results = {}
        image_files = [f for f in os.listdir(image_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

        print(f"Found {len(image_files)} images in {image_dir}")

        for img_file in image_files:
            img_path = os.path.join(image_dir, img_file)
            try:
                predictions, _ = self.predict_single(img_path, top_k)
                results[img_file] = predictions
                print(f"\n{img_file}:")
                for pred in predictions:
                    print(f"  {pred['class']}: {pred['confidence']:.2f}%")
            except Exception as e:
                print(f"Error processing {img_file}: {e}")
                results[img_file] = None

        return results

    def predict_and_visualize(self, image_path, top_k=5, save_path=None):
        import matplotlib.pyplot as plt

        predictions, image = self.predict_single(image_path, top_k)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        ax1.imshow(image)
        ax1.axis('off')
        ax1.set_title('Input Image', fontsize=14)

        classes = [pred['class'] for pred in predictions]
        confidences = [pred['confidence'] for pred in predictions]

        colors = plt.cm.viridis([i/len(confidences) for i in range(len(confidences))])
        bars = ax2.barh(range(len(classes)), confidences, color=colors)
        ax2.set_yticks(range(len(classes)))
        ax2.set_yticklabels(classes)
        ax2.set_xlabel('Confidence (%)')
        ax2.set_title('Top Predictions', fontsize=14)
        ax2.set_xlim([0, 100])

        for i, (bar, conf) in enumerate(zip(bars, confidences)):
            ax2.text(conf + 1, i, f'{conf:.1f}%', 
                    va='center', fontsize=10)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        else:
            plt.show()

        plt.close()

        return predictions


def main():
    parser = argparse.ArgumentParser(description='Flower Classification Prediction')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--image', type=str, help='Path to single image for prediction')
    parser.add_argument('--image_dir', type=str, help='Path to directory of images for batch prediction')
    parser.add_argument('--top_k', type=int, default=5, help='Number of top predictions to show')
    parser.add_argument('--visualize', action='store_true', help='Visualize predictions')
    parser.add_argument('--save_dir', type=str, default='predictions', help='Directory to save visualizations')

    args = parser.parse_args()

    if not args.image and not args.image_dir:
        print("Error: Please provide either --image or --image_dir")
        return

    predictor = FlowerPredictor(args.checkpoint)

    if args.visualize:
        os.makedirs(args.save_dir, exist_ok=True)

    if args.image:
        print(f"\nPredicting: {args.image}")
        predictions = predictor.predict_single(args.image, args.top_k)[0]
        
        print("\nTop predictions:")
        for i, pred in enumerate(predictions, 1):
            print(f"{i}. {pred['class']}: {pred['confidence']:.2f}%")

        if args.visualize:
            img_name = os.path.splitext(os.path.basename(args.image))[0]
            save_path = os.path.join(args.save_dir, f"{img_name}_prediction.png")
            predictor.predict_and_visualize(args.image, args.top_k, save_path)

    elif args.image_dir:
        results = predictor.predict_batch(args.image_dir, args.top_k)

        if args.visualize:
            for img_file, predictions in results.items():
                if predictions:
                    img_name = os.path.splitext(img_file)[0]
                    save_path = os.path.join(args.save_dir, f"{img_name}_prediction.png")
                    img_path = os.path.join(args.image_dir, img_file)
                    predictor.predict_and_visualize(img_path, args.top_k, save_path)


if __name__ == '__main__':
    main()
