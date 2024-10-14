import torch
import cv2
import numpy as np
from facenet_pytorch import MTCNN
from torchvision import transforms
import matplotlib.pyplot as plt

# Global variable for the number of frames to sample
NUM_FRAMES = 16  # Example: choose 16 evenly spaced frames from the video

# Initialize the face detector (MTCNN)
face_detector = MTCNN(keep_all=False, device='cuda' if torch.cuda.is_available() else 'cpu')

# Function to sample evenly spaced frames
def sample_frames(video_frames, num_frames):
    """
    Samples evenly spaced frames from the video.
    
    Args:
        video_frames (torch.Tensor): Input video frames of shape [num_total_frames, channels, height, width].
        num_frames (int): The number of frames to sample.
    
    Returns:
        torch.Tensor: Sampled frames.
    """
    total_frames = video_frames.shape[0]
    indices = torch.linspace(0, total_frames - 1, steps=num_frames).long()
    sampled_frames = video_frames[indices]
    return sampled_frames

# Preprocessing function to extract faces from each frame
def preprocess_video(video_frames, target_size=(224, 224)):
    """
    Preprocesses video frames by detecting and cropping faces.
    
    Args:
        video_frames (torch.Tensor): Input video frames of shape [batch_size, num_total_frames, channels, height, width].
        target_size (tuple): The size to which the cropped face is resized (default is (224, 224)).
        
    Returns:
        torch.Tensor: Preprocessed video frames with faces cropped and resized to target_size.
    """
    preprocessed_frames = []

    # Apply frame sampling to select NUM_FRAMES from the total frames
    video_frames = sample_frames(video_frames[0], NUM_FRAMES)  # Extract only NUM_FRAMES
    
    # Iterate through the sampled frames
    for t in range(video_frames.shape[0]):
        frame = video_frames[t, :, :, :].cpu().numpy()  # Get the t-th frame in numpy format
        
        # Convert frame from torch format to numpy array with channels last (HWC format)
        frame_rgb = np.transpose(frame, (1, 2, 0))
        
        # Detect face in the frame using MTCNN
        face_crop = face_detector(frame_rgb)
        
        if face_crop is not None:
            # If a face is detected, resize it to target size
            face_resized = transforms.Resize(target_size)(face_crop)
            
            # Convert the face crop to tensor and normalize it
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            face_tensor = transform(face_resized)
        else:
            # If no face is detected, return a zeroed tensor of target size (placeholder)
            face_tensor = torch.zeros((3, target_size[0], target_size[1]))
        
        preprocessed_frames.append(face_tensor)
    
    # Stack preprocessed frames along the time dimension
    preprocessed_frames = torch.stack(preprocessed_frames, dim=0).unsqueeze(0)
    
    return preprocessed_frames


# def VideoDataLoader
    
