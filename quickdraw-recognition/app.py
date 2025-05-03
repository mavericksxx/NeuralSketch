from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
import base64
import numpy as np
from PIL import Image, ImageOps
import io
import os
import logging
import sys
import cv2
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
import torch.serialization

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

WHITE_RGB = (255, 255, 255)
CLASSES = ["apple", "book", "bowtie", "candle", "cloud", "cup", "door", "envelope", "eyeglasses", "guitar", "hammer",
           "hat", "ice cream", "leaf", "scissors", "star", "t-shirt", "pants", "lightning", "tree"]

app = Flask(__name__)

class QuickDraw(nn.Module):
    def __init__(self, input_size=28, num_classes=20):
        super(QuickDraw, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Sequential(nn.Conv2d(1, 32, 5, bias=False), nn.ReLU(inplace=True), nn.MaxPool2d(2,2))
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, 5, bias=False), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2))
        dimension = int(64 * pow(input_size/4 - 3, 2))
        self.fc1 = nn.Sequential(nn.Linear(dimension, 512), nn.Dropout(0.5))
        self.fc2 = nn.Sequential(nn.Linear(512, 128), nn.Dropout(0.5))
        self.fc3 = nn.Sequential(nn.Linear(128, num_classes))

    def forward(self, input):
        output = self.conv1(input)
        output = self.conv2(output)
        output = output.view(output.size(0), -1)
        output = self.fc1(output)
        output = self.fc2(output)
        output = self.fc3(output)
        return output

def process_image_for_model(image_data):

    nparr = np.frombuffer(base64.b64decode(image_data.split(',')[1]), np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    

    ys, xs = np.nonzero(image)
    if len(ys) == 0 or len(xs) == 0:  # Empty image
        return None
        
    min_y, max_y = np.min(ys), np.max(ys)
    min_x, max_x = np.min(xs), np.max(xs)
    image = image[min_y:max_y + 1, min_x:max_x + 1]

    image = cv2.resize(image, (28, 28))
    

    image = np.array(image, dtype=np.float32)
    image = image[None, None, :, :]  
    image = torch.from_numpy(image)
    return image


MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "whole_model_quickdraw")

def load_quickdraw_model():
    try:
        torch.serialization.add_safe_globals(['QuickDraw'])
        
        model = QuickDraw(num_classes=20)

        # i know both the statements are the same, but for some reason the model wouldnt load without having an else CPU statement 
        # even though im loading it onto the gpu ...
        
        if torch.cuda.is_available():
            state_dict = torch.load(MODEL_PATH, map_location='cuda:0', weights_only=False)
        else:
            state_dict = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
        
        if isinstance(state_dict, nn.Module):
            model = state_dict
        else:
            model.load_state_dict(state_dict)
            
        model.eval()
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def init_controlnet():
    controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_scribble")
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", 
        controlnet=controlnet,
        safety_checker=None 
    )
    
    if torch.backends.mps.is_available():
        pipe = pipe.to("mps")
    elif torch.cuda.is_available():
        pipe = pipe.to("cuda")
    else:
        pipe = pipe.to("cpu")
        
    return pipe

try:
    controlnet_pipe = init_controlnet()
    print("ControlNet initialized successfully")
except Exception as e:
    print(f"Error initializing ControlNet: {str(e)}")
    controlnet_pipe = None

def process_base64_image(base64_string):
    image_data = base64.b64decode(base64_string.split(',')[1])
    return Image.open(io.BytesIO(image_data))

def image_to_base64(image):
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
        image_tensor = process_image_for_model(request.json['image'])
        if image_tensor is None:
            return jsonify({'error': 'Empty image provided'}), 400
        
        logger.debug("Loading QuickDraw model...")
        try:
            model = load_quickdraw_model()
            if torch.cuda.is_available():
                image_tensor = image_tensor.cuda()
            logger.debug("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return jsonify({'error': str(e)}), 500
        
        with torch.no_grad():
            outputs = model(image_tensor)
        
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        
        top5_prob, top5_indices = torch.topk(probabilities, 5)
        
        top5_prob = top5_prob.cpu().tolist()
        top5_indices = top5_indices.cpu().tolist()
        
        top_predictions = [
            {'class': CLASSES[idx], 'confidence': float(prob)}
            for idx, prob in zip(top5_indices, top5_prob)
        ]
        
        return jsonify({'top_predictions': top_predictions})
    
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/refine', methods=['POST'])
def refine_artwork():
    if 'image' not in request.json:
        return jsonify({'error': 'No image provided'}), 400
    
    try:
        if controlnet_pipe is None:
            return jsonify({'error': 'ControlNet not initialized'}), 500

        image = process_base64_image(request.json['image'])
        
        sketch = image.convert("L").resize((512, 512))
        
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # TO-DO:
        # UNSURE IF I WANT TO KEEP THIS :(
        # HERE IS THE DEFAULT SYSTEM PROMPT FED TO THE CONTROLNET PIPELINE
        # THIS CAN BE CHANGED BY THE USER IN THE FRONTEND
        # IF NOTHING IS PASSED IN THE PROMPT, IT WILL DEFAULT TO THIS
        # THE RESULTS ARE SUB PAR IMO, REMEMBER TO CHANGE THIS LATER TO SOMETHING BETTER
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        prompt = request.json.get('prompt', 'a highly detailed digital art masterpiece of this drawing, professional quality, perfect lighting, vibrant colors, smooth lines, artistic composition, trending on artstation, 4k resolution, professional digital illustration')
        
        with torch.inference_mode():
            output = controlnet_pipe(
                prompt=prompt,
                image=sketch,
                num_inference_steps=30,
                guidance_scale=7.5
            )
        
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