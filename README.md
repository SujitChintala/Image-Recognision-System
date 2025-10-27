# Image Recognition System with Deep Learning

A powerful image recognition system built with PyTorch and Flask that uses a pre-trained AlexNet model to classify images into 1,000 different categories from the ImageNet dataset. Features both a modern web interface and command-line interface.

## 🌟 Features

- **Web Application**: Beautiful, responsive UI with drag-and-drop image upload
- **Command-Line Interface**: Simple script for quick predictions
- **Pre-trained AlexNet Model**: Industry-standard CNN architecture
- **Top-5 Predictions**: Get confidence scores for multiple predictions
- **1,000+ Categories**: Recognizes animals, vehicles, objects, and more
- **Real-time Processing**: Fast image classification
- **Auto-cleanup**: Uploaded files are automatically removed after processing

## 🧠 About the Technology

### AlexNet
AlexNet is a pioneering convolutional neural network (CNN) architecture that won the ImageNet Large Scale Visual Recognition Challenge in 2012. It revolutionized computer vision and deep learning, achieving breakthrough accuracy in image classification across 1,000 different object categories.

## 📋 Prerequisites

- Python 3.7 or higher
- pip (Python package manager)
- Git (for cloning the repository)

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/SujitChintala/Image-Recognition-System.git
cd Image-Recognition-System
```

2. Create a new environment and activate it:
```bash
python -m venv venv
venv\Scripts\activate
```

3. Install required dependencies:
```bash
pip install -r requirements.txt
```

The required packages include:
- `torch>=1.9.0` - PyTorch deep learning framework
- `torchvision>=0.10.0` - Computer vision utilities
- `Pillow>=8.3.0` - Image processing library
- `Flask>=2.0.0` - Web framework (for web app)
- `flask-cors>=3.0.10` - Cross-origin resource sharing

## 💻 Usage

### Option 1: Web Application (Recommended)

1. Start the Flask web server:
```bash
python app.py
```

2. Open your browser and navigate to:
```
http://localhost:5000
```

3. Use the web interface to:
   - Drag and drop an image or click to upload
   - Click "Analyze Image" to get predictions
   - View top predictions with confidence scores
   - Try different images!

### Option 2: Command-Line Interface

1. Run the image recognition script:
```bash
python image_recognition.py
```

2. When prompted, enter the path to your image:
```
Enter the path to your image: path/to/your/image.jpg
```

3. The program will display the top 5 predictions with confidence scores.

## 📸 Example Output

### Web Interface
The web application provides an intuitive interface with:
- Modern, responsive design
- Real-time image preview
- Visual confidence bars
- Top prediction highlighted
- Color-coded confidence scores

### Command-Line Output
```
==================================================
Image Recognition with AlexNet
==================================================
Loading alexnet model...
Model loaded successfully!

Enter the path to your image: dog.jpg

Image: dog.jpg
--------------------------------------------------
Top Predictions:
1. golden retriever: 96.29%
2. Labrador retriever: 2.15%
3. cocker spaniel: 0.45%
4. Irish setter: 0.23%
5. English setter: 0.15%
--------------------------------------------------

✓ Best match: golden retriever (96.29% confidence)
```

## 🔧 How It Works

1. **Model Loading**: The system loads a pre-trained AlexNet model that has been trained on millions of images from the ImageNet dataset.

2. **Image Preprocessing**: Input images undergo several transformations:
   - Resized to 256×256 pixels
   - Center-cropped to 224×224 pixels
   - Converted to tensor format
   - Normalized using ImageNet mean and standard deviation values

3. **Prediction**: The preprocessed image is passed through the neural network to generate predictions across all 1,000 possible classes.

4. **Results**: The model outputs confidence scores, and the top 5 predictions are displayed with their respective confidence percentages.

5. **Cleanup** (Web App): Uploaded files are automatically deleted after processing to save storage space.

## 📁 Project Structure

```
Image-Recognition-System/
│
├── app.py                  # Flask web application
├── image_recognition.py    # Core recognition module and CLI
├── test_example.py         # Test script
├── requirements.txt        # Python dependencies
├── imagenet_classes.txt    # ImageNet class labels (auto-downloaded)
├── README.md              # Project documentation
│
├── templates/
│   └── index.html         # Web interface
│
├── uploads/               # Temporary upload folder (auto-created)
│
└── __pycache__/          # Python cache files
```

## 🖼️ Supported Image Formats

The system supports common image formats including:
- JPG/JPEG
- PNG
- BMP
- GIF

**Maximum file size**: 16MB (web application)

## ⚙️ Technical Details

- **Model Architecture**: AlexNet (pre-trained on ImageNet)
- **Framework**: PyTorch 1.9.0+
- **Web Framework**: Flask 2.0.0+
- **Input Size**: 224×224×3 (RGB images)
- **Number of Classes**: 1,000 ImageNet categories
- **Training Dataset**: ImageNet ILSVRC
- **API Endpoint**: `/predict` (POST with multipart/form-data)

## 📝 API Usage

### Web API Endpoint

**Endpoint**: `POST /predict`

**Request**:
- Content-Type: `multipart/form-data`
- Body: Image file with key `file`

**Response**:
```json
{
  "success": true,
  "predictions": [
    {"class": "golden retriever", "confidence": 96.29},
    {"class": "Labrador retriever", "confidence": 2.15},
    ...
  ],
  "top_prediction": {
    "class": "golden retriever",
    "confidence": 96.29
  }
}
```

**Error Response**:
```json
{
  "error": "Error message here"
}
```

## 💡 Notes

- **First Run**: The first time you run the program, it will automatically download:
  - Pre-trained AlexNet model weights (~240 MB)
  - ImageNet class labels (~20 KB)
- **Performance**: The model performs best on images similar to ImageNet training data
- **Best Results**: Use clear, well-lit images with the object centered
- **CORS Enabled**: The web API supports cross-origin requests
- **Auto-cleanup**: Uploaded images are automatically deleted after processing

## 🛠️ Troubleshooting

**Issue**: Model download is slow  
**Solution**: The first run downloads the model weights (~240 MB). This is a one-time process and subsequent runs will be much faster.

**Issue**: "Image file not found" error (CLI)  
**Solution**: Ensure you provide the correct path to your image file. Try using absolute paths if relative paths don't work.

**Issue**: Low confidence predictions  
**Solution**: Try using a clearer image with the object centered and well-lit. The model works best with images similar to those in the ImageNet dataset.

**Issue**: Flask app won't start  
**Solution**: Make sure port 5000 is not already in use. You can change the port in `app.py` if needed.

**Issue**: "No module named 'flask'" or similar import errors  
**Solution**: Ensure all dependencies are installed: `pip install -r requirements.txt`

**Issue**: CORS errors when accessing the API  
**Solution**: CORS is enabled by default via `flask-cors`. Ensure the package is installed.

## 🎯 Use Cases

- **Educational**: Learn about deep learning and computer vision
- **Prototyping**: Quick image classification for projects
- **Testing**: Evaluate model performance on custom images
- **Integration**: Use the API endpoint in other applications
- **Research**: Understand CNN architectures and transfer learning

## 📚 References

- [PyTorch Documentation](https://pytorch.org/docs/)
- [TorchVision Models](https://pytorch.org/vision/stable/models.html)
- [ImageNet Dataset](http://www.image-net.org/)
- [AlexNet Paper](https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html)
- [Flask Documentation](https://flask.palletsprojects.com/)

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest new features
- Submit pull requests
- Improve documentation

## 📄 License

This project is for educational purposes. The AlexNet model and ImageNet labels are used under their respective licenses.

## 👨‍💻 Author

- GitHub: [SujitChintala](https://github.com/SujitChintala) 
- LinkedIn: [Saai Sujit Chintala](https://www.linkedin.com/in/sujitchintala/)
- Email: sujitchintala@gmail.com

---

<div align="center">

**⭐ Star this repository if you find it helpful!**

</div>