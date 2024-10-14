import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import vit_b_16  # Vision Transformer from torchvision

class ViT_LSTM(nn.Module):
    def __init__(self, hidden_dim = 512, lstm_layers = 2):
        super(ViT_LSTM, self).__init__()
        
        # Load pre-trained Vision Transformer
        self.vit = vit_b_16(weights=True)
        self.vit_fc_in_features = self.vit.heads.head.in_features
        
        # Remove the final classification head of ViT
        self.vit.heads = nn.Identity()

        # LSTM layer to process the sequence of ViT outputs
        self.lstm = nn.LSTM(input_size=self.vit_fc_in_features, 
                            hidden_size=hidden_dim, 
                            num_layers=lstm_layers, 
                            batch_first=True)
        
        # Final layer for binary classification (single output for each frame)
        self.fc = nn.Linear(hidden_dim, 1)  # Single output for each frame
        
        # Sigmoid activation to output a probability between 0 and 1
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, num_frames, channels, height, width = x.shape
        
        # Initialize a tensor to store the frame features
        vit_features = []
        
        # Process each frame with the Vision Transformer
        for t in range(num_frames):
            frame = x[:, t, :, :, :]  # Extract the t-th frame
            frame_features = self.vit(frame)  # Pass frame through ViT
            vit_features.append(frame_features)
        
        # Stack the features to form a sequence [batch_size, num_frames, vit_features]
        vit_features = torch.stack(vit_features, dim=1)
        
        # Pass the sequence of features through LSTM
        lstm_out, _ = self.lstm(vit_features)
        
        # Frame-by-frame classification using LSTM outputs
        frame_predictions = self.fc(lstm_out)  # Shape: [batch_size, num_frames, 1]
        
        # Apply sigmoid activation to get outputs between 0 and 1
        frame_probabilities = self.sigmoid(frame_predictions)  # Shape: [batch_size, num_frames, 1]
        
        return frame_probabilities

# Example usage
if __name__ == "__main__":
    # Hyperparameters

    model = ViT_LSTM(hidden_dim, lstm_layers)
    
    # Example input: batch of 4 videos, each with 16 frames, 3 color channels, and 224x224 resolution
    input_videos = torch.randn(4, 16, 3, 224, 224)  # Shape: [batch_size, num_frames, channels, height, width]
    
    # Forward pass
    frame_outputs = model(input_videos)
    print(frame_outputs.shape)  # Expected shape: [batch_size, num_frames, 1] (probability per frame)
