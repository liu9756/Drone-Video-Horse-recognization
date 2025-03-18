import os
from glob import glob

class DataLoader:
    """
    Load segmented horse data from a video
    """
    def __init__(self, video_path):
        """
        init loader
        :param video_path: video path
        """
        self.video_path = video_path

    def load_data(self):
        """
        iteration on the structure, returning the data dictionary
        """
        data = {}
        # iteration on time segments
        segments = sorted([d for d in os.listdir(self.video_path)
                           if os.path.isdir(os.path.join(self.video_path, d))])
        for seg in segments:
            seg_path = os.path.join(self.video_path, seg)
            crop_path = os.path.join(seg_path, 'crop')
            if not os.path.exists(crop_path):
                print(f"Warning: {crop_path} non exist so ignore.")
                continue

            # horse 1 -7
            horses = sorted([d for d in os.listdir(crop_path)
                             if os.path.isdir(os.path.join(crop_path, d))])
            data[seg] = {}
            for horse in horses:
                horse_path = os.path.join(crop_path, horse)
                frame_files = sorted(glob(os.path.join(horse_path, '*.jpg')))
                data[seg][horse] = frame_files

        return data

if __name__ == '__main__':
    # Path for video 1 - 5
    # Video 1: /home/liu.9756/Drone_video/labeled_Dataset_DJI_0265/
    # Video 2: /home/liu.9756/Drone_video/labeled_Dataset_DJI_0266/
    # Video 3: /home/liu.9756/Drone_video/labeled_Dataset_DJI_0267/
    # Video 4: /home/liu.9756/Drone_video/labeled_Dataset_DJI_0268/
    # Video 5: /home/liu.9756/Drone_video/labeled_Dataset_DJI_0269/
    video_dir = '/home/liu.9756/Drone_video/labeled_Dataset_DJI_0265/'  
    loader = DataLoader(video_dir)
    data = loader.load_data()

    for seg, horses in data.items():
        print(f"For time: {seg}")
        for horse, frames in horses.items():
            print(f"  {horse}: {len(frames)} frames")
