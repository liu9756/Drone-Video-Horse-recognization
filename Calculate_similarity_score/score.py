import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from data_preprocess import data_preprocess
from scipy.optimize import linear_sum_assignment
import csv

def compute_similarity_matrix(ref_seq, comp_seq, sim_metric='euclidean'):
    if isinstance(ref_seq, list):
        ref_seq = np.stack(ref_seq, axis=0)
    if isinstance(comp_seq, list):
        comp_seq = np.stack(comp_seq, axis=0)
        
    if sim_metric == 'euclidean':
        diff = ref_seq[:, np.newaxis, :] - comp_seq[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        sim_matrix = 1.0 / (1.0 + dist)
    elif sim_metric == 'cosine':
        sim_matrix = np.dot(ref_seq, comp_seq.T)
        sim_matrix = (sim_matrix + 1) / 2
    return sim_matrix

def aggregate_matrix(sim_matrix, agg_method='average', weight_alpha=0.1):
    if agg_method == 'average':
        return np.mean(sim_matrix)
    elif agg_method == 'maximum':
        return np.max(sim_matrix)
    elif agg_method == 'top20':
        flat = sim_matrix.flatten()
        k = max(1, int(0.2 * len(flat)))
        top_k = np.sort(flat)[-k:]
        return np.mean(top_k)
    elif agg_method == 'top10':
        flat = sim_matrix.flatten()
        k = max(1, int(0.1 * len(flat)))
        top_k = np.sort(flat)[-k:]
        return np.mean(top_k)
    elif agg_method == 'top30':
        flat = sim_matrix.flatten()
        k = max(1, int(0.3 * len(flat)))
        top_k = np.sort(flat)[-k:]
        return np.mean(top_k)
    elif agg_method == 'top40':
        flat = sim_matrix.flatten()
        k = max(1, int(0.4 * len(flat)))
        top_k = np.sort(flat)[-k:]
        return np.mean(top_k)
    elif agg_method == 'top20_weighted':
        flat = np.sort(sim_matrix.flatten())[-max(1, int(0.2 * len(sim_matrix.flatten()))):]
        k = len(flat)
        weights = np.arange(1, k+1)
        weighted_average = np.sum(flat * weights) / np.sum(weights)
        return weighted_average
    elif agg_method == 'top20_median':
        flat = sim_matrix.flatten()
        k = max(1, int(0.2 * len(flat)))
        top_k = np.sort(flat)[-k:]
        return np.median(top_k)
    elif agg_method == 'trimmed':
        flat = np.sort(sim_matrix.flatten())
        n = len(flat)
        lower = int(0.05 * n)
        upper = int(0.95 * n)
        trimmed = flat[lower:upper]
        return np.mean(trimmed)
    elif agg_method == 'geometric':
        eps = 1e-8
        flat = sim_matrix.flatten() + eps
        return np.exp(np.mean(np.log(flat)))
    elif agg_method == 'p90':
        return np.percentile(sim_matrix, 90)
    elif agg_method == 'weighted':
        T1, T2 = sim_matrix.shape
        rows = np.arange(T1).reshape(-1, 1)
        cols = np.arange(T2).reshape(1, -1)
        weights = np.exp(-weight_alpha * np.abs(rows - cols))
        weighted_sim = np.sum(sim_matrix * weights) / np.sum(weights)
        return weighted_sim

def compute_similarity_score(ref_seq, comp_seq, sim_metric='euclidean', agg_method='average', weight_alpha=0.1):
    sim_matrix = compute_similarity_matrix(ref_seq, comp_seq, sim_metric)
    score = aggregate_matrix(sim_matrix, agg_method, weight_alpha)
    return score


def visualize_similarity(sim_matrix, ref_names, comp_names, title, save_path):
    plt.figure(figsize=(8, 6))
    im = plt.imshow(sim_matrix, interpolation='nearest', cmap='viridis')
    plt.colorbar(im)
    plt.xticks(np.arange(len(comp_names)), comp_names, rotation=45)
    plt.yticks(np.arange(len(ref_names)), ref_names)
    plt.title(title)

    for i in range(sim_matrix.shape[0]):
        for j in range(sim_matrix.shape[1]):
            plt.text(j, i, f"{sim_matrix[i, j]:.2f}",
                     ha="center", va="center", color="white", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def bipartite_match(sim_matrix):
    cost_matrix = -sim_matrix  
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return row_ind, col_ind


if __name__ == '__main__':
    video_ref_path = '/home/liu.9756/Drone_video/labeled_Dataset_DJI_0268/'   
    video_comp_path = '/home/liu.9756/Drone_video/labeled_Dataset_DJI_0266/'  

    print("Preprocessing reference...")
    processed_data_ref = data_preprocess(video_ref_path, fps=60)
    print("Preprocessing compare...")
    processed_data_comp = data_preprocess(video_comp_path, fps=60)

    segment_ref = sorted(processed_data_ref.keys())[0]
    segment_comp = sorted(processed_data_comp.keys())[0]

    ref_horses = processed_data_ref[segment_ref]
    comp_horses = processed_data_comp[segment_comp]
    
    ref_names = sorted(ref_horses.keys())
    comp_names = sorted(comp_horses.keys())
    
    agg_methods = ['average', 'maximum', 'top20', 'weighted','top10','top30','top40','top20_weighted','top20_median','trimmed','geometric','p90']
    
    similarity_results = {method: np.zeros((len(ref_names), len(comp_names))) for method in agg_methods}
    
    for i, ref_horse in enumerate(ref_names):
        ref_seq = ref_horses[ref_horse]  
        for j, comp_horse in enumerate(comp_names):
            comp_seq = comp_horses[comp_horse]
            sim_matrix = compute_similarity_matrix(ref_seq, comp_seq, sim_metric='euclidean')
            for method in agg_methods:
                score = aggregate_matrix(sim_matrix, agg_method=method, weight_alpha=0.1)
                similarity_results[method][i, j] = score

    for method in agg_methods:
        print(f"{method} Similarity matrix:")
        print(similarity_results[method])
        # bipartite matching
        row_ind, col_ind = bipartite_match(similarity_results[method])
        print(f"bipartite matching ({method}):")
        bipartite_results = []  
        for i, j in zip(row_ind, col_ind):
            score = similarity_results[method][i, j]
            match_info = (ref_names[i], comp_names[j], score)
            bipartite_results.append(match_info)
            print(f"  Ref: {ref_names[i]}  <-->  Comp: {comp_names[j]}, Score: {score:.4f}")

        csv_filename = f"bipartite_match_{method}_0268_vs_0266.csv"
        with open(csv_filename, mode='w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Reference Horse", "Comparison Horse", "Score"])
            for ref_horse, comp_horse, score in bipartite_results:
                writer.writerow([ref_horse, comp_horse, f"{score:.4f}"])
        print(f"Output csv file：{csv_filename}")
        title = f"Similarity Score Matrix ({method})"
        save_path = f"similarity_{method}__0268_vs_0266.png"
        visualize_similarity(similarity_results[method], ref_names, comp_names, title, save_path)
        print(f"Saved '{method}' similarity heatmap：{save_path}")