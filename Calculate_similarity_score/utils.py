from data_preprocess import data_preprocess
import os
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from data_loading import DataLoader
def combine_video_features(video_paths, fps=60):
    combined = {}
    for path in video_paths:
        processed_data = data_preprocess(path, fps)
        for segment in processed_data:
            for horse in processed_data[segment]:
                if horse not in combined:
                    combined[horse] = []
                combined[horse].extend(processed_data[segment][horse])
    return combined

def gather_data_from_videos(video_dirs):
    data_list = []
    horse_set = set()

    for vdir in video_dirs:
        if not os.path.isdir(vdir):
            print(f"Warning: {vdir} does not exist, skipping.")
            continue

        loader = DataLoader(vdir)
        data = loader.load_data()  

        for seg, horses in data.items():
            for horse, frames in horses.items():
                horse_set.add(horse)  
                for img_path in frames:
                    data_list.append((img_path, horse))

    return data_list, horse_set


def create_label_dict(horses):
    """
    horses: iterable of horse_names (strings)
    return: {horse_name: label_id}
    """
    horse_list = sorted(list(horses))
    return { horse_name: idx for idx, horse_name in enumerate(horse_list) }

def get_confusion_matrix(all_labels,all_preds):
    cm = confusion_matrix(all_labels, all_preds) 
    print("Confusion Matrix:\n", cm)
    return cm

def get_classification_report(all_labels,all_preds,index_to_name,digits):
    report = classification_report(
    all_labels,
    all_preds,
    target_names=index_to_name,  
    digits=3  
    )
    print(report)
    return report
