#We'll follow a specific processto build a fashion recommendation system using image features:
# 1. Assemble a diverse dataset of fashion items. This dataset should include a wide variety of items with different colours, patterns, styles, and categories. 
# 2. Ensure all the images are in a consistent format (e.g. JPEG, PNG) and resolution. 
# 3. Implement a preprocessing function to prepare images for feature extraction.
# 4. Choose a pre-trained CNN model such as VGG16, ResNet or InceptionV3. These models, pre-trained
#on large datasets like ImageNet, are capable of extracting powerful feature representations from images. 
# 5. Pass each image through the pre-trained CNN model to extract features. 
# 6. Define a metric for measuring the similarity between feature vectors.
# 7. Rank the dataset images based on the similarity to the inout image and recommend the top N items that are most similar.
# 8. Implement a final function that encapsulates the entire process from pre-processing an input image, extracting features, computing similarities, and outputting recommendations. 

# Importing the required libraries

from zipfile import ZipFile
import os

# Extracting the dataset
zip_file_path = "women-fashion.zip"
extraction_directory = "women_fashion"

if not os.path.exists(extraction_directory):
    os.makedirs(extraction_directory)

with ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extraction_directory)

# Checking the contents of the extracted directory

extracted_files = os.listdir(extraction_directory)
print(extracted_files[:10])

#We used Python's ZipFile module, the zip file is opened in read mode, and its contents are extracted to the designated directory. 

# Let's list its contents to understand the types and number of images we have:

# correcting the path to include the "women fashion" directory and listing its contents

extraction_directory_updated = os.path.join(extraction_directory, "women fashion")

# Listing the contents of the updated directory
extracted_files_updated = os.listdir(extraction_directory_updated)
print(extracted_files_updated[:10], len(extracted_files_updated))

#Let's have a look at the first image from the dataset:

from PIL import Image
import matplotlib.pyplot as plt

# function to load and display an image

def display_image(file_path):
    image = Image.open(file_path)
    plt.imshow(image)
    plt.axis('off')
    plt.show()


#Display the first image to understand its characteristics

first_image_path = os.path.join(extraction_directory_updated, extracted_files_updated[0])
display_image(first_image_path)

#Now we will create a list of all image file paths that will be used later in extracting the features from every image in the dataset:

import glob

#directory path containing all the images

image_directory = "women_fashion/women fashion"
image_paths_list = [file for file in glob.glob(os.path.join(image_directory, "*.*")) if file.endswith(('jpg', 'jpeg', 'png', 'webp'))]

#print the list of image file paths
print(image_paths_list)

#The glob module is used to generate a list of file paths for images stored in the directory. The glob.glob functionà searches for files that match a specified pattern, in this case  *.*, which matches all the files within the directory. 
#The list comprehension then filters these files to include only those with specific image file extesions. 

#It ensures that image_paths_list contains paths to only the image files, excluding any other file types that might be present in the directory.

#Now we will extract features from all the fashion images:
#I'm using # type: ignore to ignore the error, for some reason VS Code can't find the modules

from tensorflow.keras.preprocessing import image # type: ignore
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input # type: ignore
from tensorflow.keras.applications.vgg16 import preprocess_input # type: ignore
from tensorflow.keras.models import Model # type: ignore
import numpy as np

base_model = VGG16(weights='imagenet', include_top=False)
model = Model(inputs=base_model.input, outputs=base_model.output)

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array_expanded = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array_expanded)

def extract_features(model, preprocessed_img):
    features = model.predict(preprocessed_img)
    flattened_features = features.flatten()
    normalized_features = flattened_features / np.linalg.norm(flattened_features)
    return normalized_features

all_features = []
all_image_names = []

for img_path in image_paths_list:
    preprocessed_img = preprocess_image(img_path)
    features = extract_features(model, preprocessed_img)
    all_features.append(features)
    all_image_names.append(os.path.basename(img_path))

#In the code above, a feature extraction process is implemented using the VGG16 model, a popular convolutional neural network 
#pre-trained on the ImageNet dataset, to exctract visual features from images stored in image_path_list.

#Initially, the VGG16 model is loaded without its top classification layer(include_top=False), making it suitable for feature extraction rather than classification. 
#Each image path from image_paths_list is procecces through a series of steps: the image is loaded and resized to 224x224 pixels to math che VGG16 input size requirements,
#converted to a NumPy array, and preprocessed to fit the model's expected input format. 

#The preprocessed images are then fed into the VGG16 model to extract features, which are subsequently flattened and normalized to create a consistent feature vector for each image. 
#These feature vectors (all_features) and their corresponding image filenames (all:image_names) are stored, providing structured dataset for the next steps in building a fashion recommendation system using image features.

#Now we'll write a function to recommend fashion images based on image features:

from scipy.spatial.distance import cosine # type: ignore

def recommend_fashion_items_cnn(input_image_path, all_features, all_image_names, model, top_n=5):
    # pre-process the input image and extract features
    preprocessed_img = preprocess_image(input_image_path)
    input_features = extract_features(model, preprocessed_img)

    # calculate similarities and find the top N similar images
    similarities = [1 - cosine(input_features, other_feature) for other_feature in all_features]
    similar_indices = np.argsort(similarities)[-top_n:]

    # filter out the input image index from similar_indices
    similar_indices = [idx for idx in similar_indices if idx != all_image_names.index(input_image_path)]

    # display the input image
    plt.figure(figsize=(15, 10))
    plt.subplot(1, top_n + 1, 1)
    plt.imshow(Image.open(input_image_path))
    plt.title("Input Image")
    plt.axis('off')

    # display similar images
    for i, idx in enumerate(similar_indices[:top_n], start=1):
        image_path = os.path.join('women_fashion/women fashion', all_image_names[idx])
        plt.subplot(1, top_n + 1, i + 1)
        plt.imshow(Image.open(image_path))
        plt.title(f"Recommendation {i}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

#In the code above we defined a function recommend_fashion_items_cnn, which recommends
# fashion items similar to a given input image using deep learning-based feature extraction.
# It utilizes the VGG16 model to extract high-dimensional feature vectors from images, capturing their visual essence. 

#For a specified input image, the function pre processes the image, extracts its features, and calculates the cosine similarity between
#this feature vector and those of other images in the dataset (all_features). It ranks these images based on similarity
# and selects the top N most similar images to recommend, explicitly excluding the input image from being recommended to itself
#by filtering out its index from the list of similar indices. 

#In the end, the function will visualize the input image and its recommendations by displaying them.

#Now here's how we use this function to recommend images based on a similar fashion in the input image:

input_image_path = "women_fashion/women fashion/dark, elegant, sleeveless dress that reaches down to about mid-calf.jpg"
recommend_fashion_items_cnn(input_image_path, all_features, image_paths_list, model, top_n=5)

