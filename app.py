from flask import Flask, request, render_template, jsonify
import torch, warnings
warnings.filterwarnings('ignore')
from torchvision import transforms
from PIL import Image

if torch.backends.mps.is_available():
    print("Running on MPS device")
    device = torch.device('mps')
elif torch.cuda.is_available():
    print("Running on CUDA device")
    device = torch.device('cuda')
else:
    print("Running on CPU")

app = Flask(__name__)

# Load trained model
model = torch.load('brain_tumor_mobilenet_classifier.pt',map_location=device)
model.eval()  # Set model to evaluation mode

labels_dict = {'glioma_tumor': 0, 'meningioma_tumor': 1, 'no_tumor': 2, 'pituitary_tumor': 3}
labels_dict = {v: k for k, v in labels_dict.items()}

# Define image preprocessing transformations
img_transforms = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.CenterCrop((128,128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize image to match model's input requirements
])
@app.route("/")
def home():
    return render_template('index.html')

@app.route("/predict",methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        if 'image' not in request.files:
            return jsonify({'error': 'No image file uploaded'}), 400
        # Get uploaded image file
        image_file = request.files['image']

        if image_file.filename == '':
            return jsonify({'error': 'No image file selected'}), 400
        
        try:
            image = Image.open(image_file).convert('RGB')
            # Open and preprocess the image
            transformed_img = img_transforms(image).unsqueeze(0).to(device)

            # Make prediction using the trained model
            with torch.no_grad():
                output = model(transformed_img)
                prediction = torch.max(output,1)[1]
                predicted_label = labels_dict[prediction.item()]

            predicted_label = str(' '.join(str(predicted_label).split('_'))).title()
            return render_template('index.html',prediction=predicted_label)
        except Exception as e:
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8000)