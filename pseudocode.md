# Data Preparation
FOR each category in selected_categories:
    LOAD all sketch images for category
    PREPROCESS images (resize, normalize, convert to grayscale if needed)
    SPLIT images into training and validation sets
END FOR

# Model Training (EfficientNet-B0)
INITIALIZE EfficientNet-B0 model (pretrained on ImageNet)
REPLACE final layer with new layer for num_categories

FOR epoch in range(num_epochs):
    FOR each batch in training set:
        INPUT batch_images, batch_labels
        PREDICT outputs = model(batch_images)
        CALCULATE loss (outputs, batch_labels)
        BACKPROPAGATE and update model weights
    END FOR

    EVALUATE model on validation set
    IF validation loss improves:
        SAVE model checkpoint
    END IF

    IF validation loss increases for patience epochs:
        STOP early (early stopping)
    END IF
END FOR

# Model Training (ControlNet)
INITIALIZE ControlNet model (pretrained weights)
FOR epoch in range(num_epochs):
    FOR each batch in training set:
        INPUT batch_sketches, batch_targets
        GENERATE images = ControlNet(batch_sketches)
        CALCULATE loss (images, batch_targets)
        BACKPROPAGATE and update model weights
    END FOR
END FOR

# Webapp with inference pipeline
START web server (Flask or Streamlit)

ON user drawing submission:
    GET sketch image from web canvas
    PREPROCESS image (resize, normalize)

    IF user selects "Recognize":
        LOAD EfficientNet-B0 model
        PREDICT category_probs = model(sketch_image)
        DISPLAY top-k predicted categories with confidence scores

    IF user selects "Generate Image":
        LOAD ControlNet model
        GENERATE detailed_image = ControlNet(sketch_image)
        DISPLAY generated image to user

    IF user selects "Clear":
        RESET canvas
END ON

# Visualization and Evaluation
FOR each epoch:
    RECORD training and validation loss/accuracy

AFTER training:
    COMPUTE confusion matrix on validation set
    PLOT confusion matrix and accuracy curves
    DISPLAY results in presentation or web app
