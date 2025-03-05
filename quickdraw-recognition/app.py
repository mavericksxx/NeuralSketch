from flask import Flask, render_template, request, jsonify
import torch
import base64
import numpy as np
from PIL import Image
import io
from transformers import EfficientNetForImageClassification, AutoImageProcessor
import os
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel

app = Flask(__name__)

# Load the model and processor
MODEL_PATH = "../model"  # Path relative to where app.py is located

# Initialize Stable Diffusion ControlNet
def init_controlnet():
    controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_scribble")
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", 
        controlnet=controlnet,
        safety_checker=None  # Disable safety checker for faster inference
    )
    
    # Proper device handling for Apple Silicon
    if torch.backends.mps.is_available():
        pipe = pipe.to("mps")
    elif torch.cuda.is_available():
        pipe = pipe.to("cuda")
    else:
        pipe = pipe.to("cpu")
        
    return pipe

# Initialize ControlNet when the app starts
try:
    controlnet_pipe = init_controlnet()
    print("ControlNet initialized successfully")
except Exception as e:
    print(f"Error initializing ControlNet: {str(e)}")
    controlnet_pipe = None

# Check if the model exists, if not provide instructions
if not os.path.exists(MODEL_PATH):
    print("=" * 80)
    print("Model not found. Please train the model first using these commands:")
    print("cd /Users/maverick/Developer/FDS-project/quickdraw-recognition")
    print("python download_and_preprocess.py")
    print("python train_model.py --num_epochs 5")
    print("=" * 80)

def process_base64_image(base64_string):
    """Convert base64 image data to PIL Image"""
    image_data = base64.b64decode(base64_string.split(',')[1])
    return Image.open(io.BytesIO(image_data))

def image_to_base64(image):
    """Convert PIL Image to base64 string"""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode()}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.json:
        return jsonify({'error': 'No image provided'}), 400
    
    try:
        # Get the image data
        image = process_base64_image(request.json['image'])
        
        # Load model and processor if we have a trained model
        if not os.path.exists(MODEL_PATH):
            return jsonify({
                'error': 'Model not found. Please train the model first.',
                'top_predictions': [
                    {'class': 'Example class 1', 'confidence': 0.5},
                    {'class': 'Example class 2', 'confidence': 0.3},
                ]
            })
        
        processor = AutoImageProcessor.from_pretrained(MODEL_PATH)
        model = EfficientNetForImageClassification.from_pretrained(MODEL_PATH)
        
        # Set model to evaluation mode
        model.eval()
        
        # Process the image
        inputs = processor(images=image, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Get predictions
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)[0]
        
        # Load class names
        with open(os.path.join(MODEL_PATH, "class_names.txt"), "r") as f:
            class_names = [line.strip() for line in f.readlines()]
        
        # Get top 5 predictions
        top5_prob, top5_indices = torch.topk(probabilities, 5)
        
        # Convert to Python lists
        top5_prob = top5_prob.tolist()
        top5_indices = top5_indices.tolist()
        
        # Create prediction list
        top_predictions = [
            {'class': class_names[idx], 'confidence': float(prob)}
            for idx, prob in zip(top5_indices, top5_prob)
        ]
        
        return jsonify({'top_predictions': top_predictions})
    
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/refine', methods=['POST'])
def refine_artwork():
    if 'image' not in request.json:
        return jsonify({'error': 'No image provided'}), 400
    
    try:
        # Check if ControlNet is initialized
        if controlnet_pipe is None:
            return jsonify({'error': 'ControlNet not initialized'}), 500

        # Get the image and process it
        image = process_base64_image(request.json['image'])
        
        # Convert to grayscale and resize for ControlNet
        sketch = image.convert("L").resize((512, 512))
        
        # Get the prompt (either from request or use default)
        prompt = request.json.get('prompt', 'a refined digital painting of this sketch')
        
        # Generate refined artwork
        with torch.inference_mode():
            output = controlnet_pipe(
                prompt=prompt,
                image=sketch,
                num_inference_steps=30,
                guidance_scale=7.5
            )
        
        # Convert the output image to base64
        refined_image = output.images[0]
        refined_image_b64 = image_to_base64(refined_image)
        
        return jsonify({
            'refined_image': refined_image_b64,
            'prompt': prompt
        })
        
    except Exception as e:
        print(f"Error in artwork refinement: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)