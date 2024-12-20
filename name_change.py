import os

def rename_frames(directory):
    for filename in os.listdir(directory):
        if filename.startswith("frame_") and filename.endswith(".jpg"):
            new_name = f"{int(filename.split('_')[1].split('.')[0])}.jpg"
            os.rename(os.path.join(directory, filename), os.path.join(directory, new_name))
            print(f"Renamed: {filename} -> {new_name}")

if __name__ == "__main__":
    directory = "/home/liu.9756/Drone_video/video_frames"
    rename_frames(directory)
