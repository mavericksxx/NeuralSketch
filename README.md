# NeuralSketch

## Project Structure
The application consists of three main components:
1. Data processing script (download_and_preprocess.py) - Downloads and prepares the QuickDraw dataset
2. Model training script (train_model.py) - Fine-tunes the EfficientNet-B0 model on the processed data
3. Web application (app.py and templates) - Provides a drawing interface and makes predictions

How to Run the Application
Step 1: Install Dependencies
First, install the required packages:
```
pip install -r requirements.txt
```

Step 2: Download and Preprocess the Data
Run the data download and preprocessing script:
```
python download_and_preprocess.py
```

This will:
* Download the QuickDraw dataset (.npy files) for 10 default categories
* Convert the bitmap data to images suitable for EfficientNet
* Split the data into training and testing sets
* Resize images to 224x224 pixels (EfficientNet input size)

You can customize the categories by passing the --categories argument:
```
python download_and_preprocess.py --categories cat dog house
```

Step 3: Train the Model
Train the EfficientNet-B0 model on the processed data:
```
python train_model.py --num_epochs 20 --learning_rate 0.0001
```

The script will:
* Load the EfficientNet-B0 model from Hugging Face
* Fine-tune it on the QuickDraw dataset
* Generate training curves and evaluation metrics
* Save the trained model to the model directory

Step 4: Run the Web Application
Navigate to the web app directory and start the Flask server:
```
cd quickdraw-recognition
python app.py
```

Then open your browser to http://127.0.0.1:5000 to use the application!


Using the Web Application
1. Draw an object on the canvas using your mouse or touchscreen
2. Click "Recognize Drawing" to get predictions
3. The app will display the top 5 predicted categories with confidence scores
4. Use "Clear Canvas" to start over

Notes
* By default, the application downloads data for 10 categories: cat, dog, car, apple, airplane, banana, house, flower, tree, umbrella
* The model is trained to recognize these categories, so try drawing objects from these categories for best results
* The drawing interface works on both desktop and mobile devices
* If you want to add more categories, modify the DEFAULT_CATEGORIES variable in download_and_preprocess.py and train the model again