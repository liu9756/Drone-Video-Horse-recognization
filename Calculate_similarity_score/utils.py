from data_preprocess import data_preprocess
import os
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

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
    """
    video_dirs: [path_to_video1, path_to_video2, ...]
    horse1, horse2, ...。
    return: data_list = [(img_path, horse_name), ...]
           horse_set = set([horse_name1, horse_name2, ...])
    """
    data_list = []
    horse_set = set()

    for vdir in video_dirs:
        if not os.path.isdir(vdir):
            continue
        horses = sorted(os.listdir(vdir))
        for hname in horses:
            hpath = os.path.join(vdir, hname)
            if not os.path.isdir(hpath):
                continue
            # horse hname
            horse_set.add(hname)

            # collect images
            for fname in os.listdir(hpath):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    fpath = os.path.join(hpath, fname)
                    data_list.append((fpath, hname))
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
