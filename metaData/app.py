import hashlib
import json
import subprocess
import logging

try:
    import ffmpeg
except ImportError:
    ffmpeg = None

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def hash_video(video_path, algorithm='sha256'):
    hash_obj = hashlib.new(algorithm)
    with open(video_path, 'rb') as file:
        for chunk in iter(lambda: file.read(4096), b""):
            hash_obj.update(chunk)
    return hash_obj.hexdigest()

def embed_metadata_ffmpeg_python(input_video, output_video, metadata):
    if ffmpeg is None:
        raise ImportError("ffmpeg-python is not installed. Use 'pip install ffmpeg-python' to install it.")
    
    metadata_json = json.dumps(metadata)
    
    try:
        stream = ffmpeg.input(input_video)
        stream = ffmpeg.output(stream, output_video, metadata=metadata_json)
        
        # Add verbose logging
        logger.debug(f"FFmpeg command: {ffmpeg.compile(stream)}")
        
        ffmpeg.run(stream, capture_stdout=True, capture_stderr=True)
    except ffmpeg.Error as e:
        logger.error(f"FFmpeg stderr:\n{e.stderr.decode()}")
        raise

def embed_metadata_subprocess(input_video, output_video, metadata):
    metadata_json = json.dumps(metadata)
    
    command = [
        'ffmpeg',
        '-i', input_video,
        '-metadata', f'metadata={metadata_json}',
        '-codec', 'copy',
        '-y',  # Overwrite output file if it exists
        output_video
    ]
    
    logger.debug(f"FFmpeg command: {' '.join(command)}")
    
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        logger.debug(f"FFmpeg stdout:\n{result.stdout}")
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg stderr:\n{e.stderr}")
        raise

def main():
    input_video = '/Users/hiteshgupta/Documents/SIH24/ML_Models/a8d8e9f8-783e-4078-a5e7-dc9063033031.MP4'
    output_video = '/Users/hiteshgupta/Documents/SIH24/ML_Models/out1.mp4'
    
    video_hash = hash_video(input_video)
    logger.info(f"Video hash: {video_hash}")
    
    metadata = {
        'video_hash': video_hash,
        'custom_boolean': 'true',
        'other_metadata': 'Any other information you want to include'
    }
    
    try:
        if ffmpeg is not None:
            embed_metadata_ffmpeg_python(input_video, output_video, metadata)
        else:
            embed_metadata_subprocess(input_video, output_video, metadata)
        logger.info(f"Metadata embedded. Output video: {output_video}")
    except Exception as e:
        logger.error(f"Error embedding metadata: {e}")
        return
    
    # Verification step (omitted for brevity, but you should add logging here too)

if __name__ == "__main__":
    main()