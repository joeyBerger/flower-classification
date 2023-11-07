import sys
import torch
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import numpy as np
import json
from get_cli_args import get_predict_cli_args
from PIL import Image

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    image.thumbnail((256, 256))
    
    width, height = image.size

    # Calculate the dimensions for the center crop
    size = 224
    left = (width - size) / 2
    top = (height - size) / 2
    right = (width + size) / 2
    bottom = (height + size) / 2

    # Crop the image using the calculated dimensions
    image = image.crop((left, top, right, bottom))

    means = np.array([0.485, 0.456, 0.406])
    stds = np.array([0.229, 0.224, 0.225])
    
    numpy_array = np.array(image)
    
#     numpy_array = (numpy_array - means) / stds
    numpy_array = (numpy_array / 255 - means) / stds

    # Reorder the dimensions to match PyTorch's expectation (from HWC to CHW)
    return torch.from_numpy(numpy_array.transpose((2, 0, 1))) 

def predict(image_path, model, topk=5):

    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.eval()

    img = process_image(Image.open(image_path))
    
    with torch.no_grad():
        img = img.to(device)
        output = model.forward(torch.unsqueeze(img.float(), 0))

    ps = torch.exp(output)
    
    top_p, top_class = ps.topk(5, dim=1)
    
    return top_p, top_class

def get_categories_keys(classes):
    keys = []
    for item in classes[0]:
        keys.append(str(item.item() + 1))
    return keys

def get_categories_names(keys):
    return [cat_to_name[key] for key in keys]



# Get user cli inputs
image_path, checkpoint_path, category_names, enable_gpu = get_predict_cli_args()

state_dict = torch.load(checkpoint_path)

model = models.densenet121()
model.load_state_dict(state_dict)


probs, classes = predict(image_path, model)

categories_keys = get_categories_keys(classes)

categories_names = get_categories_names(categories_keys)

fig, ax = plt.subplots()

predicted_name = categories_names[0]

ax.imshow(Image.open(image_path))  # Use 'cmap' for grayscale images

# Add title
ax.set_title(predicted_name)

plt.xticks([])
plt.yticks([])

# Show the plot
plt.show()

percentages = probs.squeeze().cpu().numpy()
percentages = np.flip(percentages)
categories_names.reverse()

# Create a horizontal bar plot
plt.barh(np.arange(len(categories_names)), percentages, color='skyblue')

# Set the y-axis labels as category names
plt.yticks(np.arange(len(categories_names)), categories_names)

# Adjust the plot layout
plt.tight_layout()

# Show the plot
plt.show()

