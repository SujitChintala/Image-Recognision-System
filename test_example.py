"""
Simple example script to test image recognition
You can modify the image_path variable to test with different images
"""

from image_recognition import ImageRecognition

def test_image_recognition():
    """Test the image recognition model"""
    
    # Initialize the model
    print("Initializing Image Recognition Model...")
    recognizer = ImageRecognition(model_name='alexnet')
    
    # Example: You can change this to your image path
    # image_path = "dog.jpg"
    # image_path = "cat.png"
    # image_path = "car.jpg"
    
    print("\nTo test the model:")
    print("1. Place an image in the project folder")
    print("2. Update the 'image_path' variable in this script")
    print("3. Uncomment the prediction code below")
    print("\nExample:")
    print("  image_path = 'your_image.jpg'")
    print("  results = recognizer.predict(image_path, top_k=5)")
    print("  recognizer.display_results(image_path, results)")
    
    # Uncomment below to test with your image
    # image_path = "your_image.jpg"
    # results = recognizer.predict(image_path, top_k=5)
    # recognizer.display_results(image_path, results)
    # top_class, top_confidence = results[0]
    # print(f"\n✓ Best match: {top_class} ({top_confidence:.2f}% confidence)")


if __name__ == "__main__":
    test_image_recognition()
