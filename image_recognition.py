"""
Image Recognition with Machine Learning using PyTorch and AlexNet
This script performs image recognition using a pre-trained AlexNet model.
"""

import torch
from torchvision import models, transforms
from PIL import Image
#import json

class ImageRecognition:
    def __init__(self, model_name='alexnet'):
        """Initialize the image recognition model"""
        print(f"Loading {model_name} model...")
        
        # Load pre-trained AlexNet model
        if model_name == 'alexnet':
            self.model = models.alexnet(pretrained=True)
        else:
            raise ValueError("Only 'alexnet' is supported in this implementation")
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Define preprocessing pipeline
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Load ImageNet class labels
        self.labels = self.load_labels()
        
        print("Model loaded successfully!")
    
    def load_labels(self):
        """Load ImageNet class labels"""
        try:
            with open('imagenet_classes.txt', 'r') as f:
                labels = [line.strip() for line in f.readlines()]
            return labels
        except FileNotFoundError:
            print("Warning: imagenet_classes.txt not found. Downloading...")
            self.download_labels()
            with open('imagenet_classes.txt', 'r') as f:
                labels = [line.strip() for line in f.readlines()]
            return labels
    
    def download_labels(self):
        """Download ImageNet labels if not present"""
        import urllib.request
        url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
        urllib.request.urlretrieve(url, 'imagenet_classes.txt')
        print("Labels downloaded successfully!")
    
    def predict(self, image_path, top_k=5):
        """
        Predict the class of an input image
        
        Args:
            image_path: Path to the input image
            top_k: Number of top predictions to return
            
        Returns:
            List of tuples (class_name, confidence_percentage)
        """
        # Load and preprocess image
        img = Image.open(image_path)
        img_tensor = self.preprocess(img)
        
        # Add batch dimension
        batch_tensor = torch.unsqueeze(img_tensor, 0)
        
        # Make prediction
        with torch.no_grad():
            output = self.model(batch_tensor)
        
        # Get probabilities
        probabilities = torch.nn.functional.softmax(output, dim=1)[0] * 100
        
        # Get top k predictions
        top_prob, top_indices = torch.topk(probabilities, top_k)
        
        results = []
        for i in range(top_k):
            class_name = self.labels[top_indices[i]]
            confidence = top_prob[i].item()
            results.append((class_name, confidence))
        
        return results
    
    def display_results(self, image_path, results):
        """Display prediction results"""
        print(f"\nImage: {image_path}")
        print("-" * 50)
        print(f"Top Predictions:")
        for i, (class_name, confidence) in enumerate(results, 1):
            print(f"{i}. {class_name}: {confidence:.2f}%")
        print("-" * 50)


def main():
    """Main function to run image recognition"""
    print("=" * 50)
    print("Image Recognition with AlexNet")
    print("=" * 50)
    
    # Initialize the model
    recognizer = ImageRecognition(model_name='alexnet')
    
    # Get image path from user
    image_path = input("\nEnter the path to your image: ").strip()
    
    try:
        # Make prediction
        results = recognizer.predict(image_path, top_k=5)
        
        # Display results
        recognizer.display_results(image_path, results)
        
        # Display the top prediction
        top_class, top_confidence = results[0]
        print(f"\n✓ Best match: {top_class} ({top_confidence:.2f}% confidence)")
        
    except FileNotFoundError:
        print(f"Error: Image file '{image_path}' not found!")
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
