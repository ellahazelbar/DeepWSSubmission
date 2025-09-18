import torch
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.keypoint_bilstm import KeypointBiLSTM
from src.data.piper import record_and_process_video, KEYPOINTS_SIZE
from predict import VOCAB, MODEL_PATH

def predict_sign(frames, model_path, num_classes=3):
    """
    Predict the ASL sign from a video file
    
    Args:
        video_path (str): Path to the input video file
        model_path (str): Path to the trained model weights
        num_classes (int): Number of classes in the model
    
    Returns:
        str: Predicted sign (letter)
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = KeypointBiLSTM(num_classes, KEYPOINTS_SIZE).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    frames = torch.FloatTensor([frames]).transpose(1,2).to(device)
    # Get prediction
    with torch.no_grad():
        outputs = model(frames)
        _, predicted = outputs.max(1)
    
    return VOCAB[predicted.item()]

def main():
    print(predict_sign(record_and_process_video(), MODEL_PATH))

if __name__ == '__main__':
    main() 