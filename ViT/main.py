# Import the required libraries.
import os
import cv2
# import pafy
import math
import random
import numpy as np
import datetime as dt
import tensorflow as tf
from collections import deque
import matplotlib.pyplot as plt

# from moviepy.editor import *

from sklearn.model_selection import train_test_split

from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model
from vit_keras import vit

print(to_categorical([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))

IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64

# Increase to increase accuracy, will also increase memory usage
SEQUENCE_LENGTH = 80

DATASET_DIR = "Celeb-DF-v2"

CLASSES_LIST = ["RealVid", "SyntheticVid"]  

# gpu_devices = tf.config.experimental.list_physical_devices('GPU')
# for device in gpu_devices:
#     tf.config.experimental.set_memory_growth(device, True)


# Frames Extractor Function

def frames_extractor(video_path):
    frames_list = []
    
    video_reader = cv2.VideoCapture(video_path)
    
    # Get the total number of frames in the video.
    video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate the the interval after which frames will be added to the list.
    skip_frames_window = max(int(video_frames_count/SEQUENCE_LENGTH), 1)
    
    for frame_counter in range(SEQUENCE_LENGTH):
         # Set the current frame position of the video.
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)

        # Reading the frame from the video. 
        success, frame = video_reader.read() 

        # Check if Video frame is not successfully read then break the loop
        if not success:
            break

        # Resize the Frame to fixed height and width.
        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
        
        # Normalize the resized frame by dividing it with 255 so that each pixel value then lies between 0 and 1
        normalized_frame = resized_frame / 255
        
        # Append the normalized frame into the frames list
        frames_list.append(normalized_frame)
    
    video_reader.release()
    
    return frames_list


# Create a function to create the dataset

def create_dataset():
    
    features = []
    labels = []
    video_files_paths = []
    
    # Iterating through all the classes mentioned in the classes list
    for class_index, class_name in enumerate(CLASSES_LIST):
        
        # Display the name of the class whose data is being extracted.
        print(f'Extracting Data of Class: {class_name}')
        
        # Get the list of video files present in the specific class name directory.
        files_list = os.listdir(os.path.join(DATASET_DIR, class_name))
        
        # Iterate through all the files present in the files list.
        for file_name in files_list:
            
            # Get the complete video path.
            video_file_path = os.path.join(DATASET_DIR, class_name, file_name)

            # Extract the frames of the video file.
            frames = frames_extractor(video_file_path)

            # Check if the extracted frames are equal to the SEQUENCE_LENGTH specified above.
            # So ignore the videos having frames less than the SEQUENCE_LENGTH.
            if len(frames) == SEQUENCE_LENGTH:

                # Append the data to their repective lists.
                features.append(frames)
                labels.append(class_index)
                video_files_paths.append(video_file_path)

    # Converting the list to numpy arrays
    features = np.asarray(features)
    labels = np.array(labels)  
    
    # Return the frames, class index, and video file path.
    return features, labels, video_files_paths

features, labels, video_files_paths = create_dataset()

# Hot encoding the labels
# # Using Keras's to_categorical method to convert labels into one-hot-encoded vectors
# one_hot_encoded_labels = 


# Split the dataset into training and testing sets.

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.25, shuffle=True)

vit_model = vit.vit_b16(
        image_size = 64,
        activation = 'softmax',
        pretrained = True,
        include_top = False,
        pretrained_top = False,
        classes = 2)

# model.summary()
def create_video_classification_model(sequence_length):
    
    # Input for video frames
    inputs = tf.keras.Input(shape=(sequence_length, 64, 64, 3))

    # Process each frame through ViT
    x = tf.keras.layers.TimeDistributed(vit_model)(inputs)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(x)
    
    # LSTM to handle the sequence of frames
    x = tf.keras.layers.LSTM(128, return_sequences=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(128, activation=tfa.activations.gelu)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(64, activation=tfa.activations.gelu)(x)
    x = tf.keras.layers.Dense(32, activation=tfa.activations.gelu)(x)
    outputs = tf.keras.layers.Dense(1, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs, name='video_classification_model')
    return model

# Create and summarize the model
model = create_video_classification_model(80)
model.summary()

early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, mode = 'min')
# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# Train the model
model_training_history = model.fit(features_train, labels_train, shuffle = True, batch_size=4, epochs=50, validation_split=0.2, callbacks=[early_stopping_callback])

# Evaluate the Model.
model_evaluation_history = model.evaluate(features_test, labels_test)

# Save the Model.
model.save("ViT/models/ViT_DF_80SL_50ep.h5")

# Plot the Training History.
plt.plot(model_training_history.history['accuracy'])