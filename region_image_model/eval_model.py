"""Eval Model"""

import collections
import numpy as np
import os
import pandas as pd
from sklearn.decomposition import PCA
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import PrecisionRecallDisplay
import tensorflow as tf
import tensorflow_datasets as tfds
from tqdm import tqdm

CONTIG_IDX = 0
START_IDX = 1
LEN_IDX = 2
LABEL_IDX = 3
DISTANCE_IDX = 4
ID_IDX = 5

CONTIG_NAME = "contig"
START_NAME = "start"
LEN_NAME = "len"
LABEL_NAME = "genotype"
DT_NAME = "distance"
ID_NAME = "id"


def save_pred(cfg, model, val_dataset, variant_info, have_id: bool = False):
    """
    Store the model, validation dataset and model's prediction into local file path specified in cfg.
    Files saved by this function are:
    
    1. "pred_results.csv" with the columns:
        Contig
        Start
        Length
        Genotype (hom.ref / otherwise)
        Predicted Distance
        ID (only for HG002)

    2. anchor_embedding, sim_embedding (optional) contain each region's embedding vector. 

    3. anchor.npy, sim.npy (optional) contain the (100, 300, 6) images of each region.
        
    Args:
        cfg: Dict a dictionary of configurations.
        model: Keras Model. model to save results of.
        val_dataset: tf.data.Dataset. dataset to run model.predict on.
        variant_info: tf.data.Dataset. dataset containing addtional meta data about the
          variant.
        have_id: (Optional) boolean defaults to False. if variant_info contain the variant id field.
    """
    # make sure local storage path exist
    os.makedirs(cfg.mode.result_data_dir, exist_ok=True)
    
    def filter_image(image, labels):
        return image["anchor"], image["sim"]

    def filter_info(image, labels):
        output = labels["contig"], labels["start"], labels["svlen"], labels["label"]
        if have_id:
            output = output + (labels["id"],)
        return output

    if cfg.mode.save_embedding:
        distance_pred, anchor_embd, sim_embd = model.predict(val_dataset)
        anchor_data_path = os.path.join(cfg.mode.result_data_dir, "anchor_embedding")
        sim_data_path = os.path.join(cfg.mode.result_data_dir, "sim_embedding")
        with open(anchor_data_path,"wb") as f: pickle.dump(anchor_embd, f)
        with open(sim_data_path,"wb") as f: pickle.dump(sim_embd, f)
        print(f"Embeddings saved at {anchor_data_path} and {sim_data_path}")
    elif cfg.mode.save_image:
        variant_img = tfds.as_numpy(variant_info.map(filter_image))
        anchor_arr, sim_arr = [], []
        for x in variant_img:
            anchor_arr.append(x[0])
            sim_arr.append(x[1])
    
        anchor_path = os.path.join(cfg.mode.result_data_dir, "anchor.npy")
        sim_path = os.path.join(cfg.mode.result_data_dir, "sim.npy")
        np.save(anchor_path, anchor_arr, allow_pickle=True)
        np.save(sim_path, sim_arr, allow_pickle=True)
        print(f"Images saved at {anchor_path} and {sim_path}")

    # convert distance output into array
    distance_pred, _, _ = model.predict(val_dataset, verbose=1, use_multiprocessing=True)
    pred_distance_arr = [x for x in distance_pred]

    # parse variant_info to a pandas dataframe
    variant_info = tfds.as_numpy(variant_info.map(filter_info))
    variant_mp = collections.defaultdict(list)
    for x in tqdm(variant_info, desc="extracting variant information", disable=False):
        variant_mp[CONTIG_NAME].append(x[0][0].decode("UTF-8")) #int decode
        variant_mp[START_NAME].append(int(x[1][0]))
        variant_mp[LEN_NAME].append(int(x[2][0]))
        variant_mp[LABEL_NAME].append(int(x[3][0]))
        if have_id:
            variant_mp[ID_NAME].append(x[4].decode("UTF-8"))

    variant_mp[DT_NAME] = pred_distance_arr
    df = pd.DataFrame(variant_mp)
    result_data_path = os.path.join(cfg.mode.result_data_dir, "pred_results.csv")
    df.to_csv(result_data_path, index=False)
    print(f"Results saved at {result_data_path}")


def plot_pca_fig(results, result_data_dir, result_dir):
    """Plot the 2-d visualization of the join embedding space by using PCA on the
    anchor and simulation embedding vectors. 

    Args:
      results: pandas Dataframe containing the eval results.
      result_data_dir: string location of the variant data.
      result_dir: string location to save the output figure.
    """
    anchor_embd_path = os.path.join(result_data_dir, "anchor_embedding")
    with open(anchor_embd_path,"rb") as f:
        anchor_embd = pickle.load(f)

    sim_embd_path = os.path.join(result_data_dir, "sim_embedding")
    with open(sim_embd_path,"rb") as f:
        sim_embd = pickle.load(f)

    pca = PCA(n_components=2)
    pca_anchor = pca.fit_transform(anchor_embd)
    anchor_numpy = np.array(pca_anchor)

    pca_sim = pca.fit_transform(sim_embd)
    sim_numpy = np.array(pca_sim)

    label = np.atleast_2d(results[LABEL_IDX]).T
    pca_arr = np.append(anchor_numpy, label, 1) # each row: xpos, ypos, label
    fig, ax = plt.subplots()
    
    colors = ['#708090', '#A52A2A', '#CCCC00']
    dot_labels = ["hom. ref", "not hom. ref", "simulation"]
    i = 0
    for xpos, ypos, label in pca_arr:
        if int(label) == 0:
            ax.scatter(float(xpos), float(ypos), c=colors[int(label)], label=dot_labels[int(label)] if i == 0 else "", alpha=0.6)
            i += 1
    
    i = 0
    for xpos, ypos, label in pca_arr:
        if int(label) == 1:
            ax.scatter(float(xpos), float(ypos), c=colors[int(label)], label=dot_labels[int(label)] if i == 0 else "", alpha=0.6)
            i += 1

    # plot sim
    i = 0
    for xpos, ypos in sim_numpy:
        ax.scatter(float(xpos), float(ypos), c=colors[2], label=dot_labels[2] if i == 0 else "", alpha=0.6)
        i += 1

    row = 0
    for i in range(len(pca_arr)):
        ax.annotate(str(row), (float(pca_arr[i][0]), float(pca_arr[i][1])))
        ax.annotate(str(row), (float(sim_numpy[i][0]), float(sim_numpy[i][1])))
        row += 1

    ax.legend()
    ax.set_title('PCA Visual')
    ax.set_ylabel('pc1')
    ax.set_xlabel('pc2')
    plot_path = os.path.join(result_dir, "pca.png")
    plt.savefig(plot_path)
    print(f"plot saved at {plot_path}")


def th_search(results, result_dir, lower: int = 0.01, upper: int = 0.5, step: int = 0.01):
    """Visualize the distribution of distance thresholds against the number of correct answer produced against
    the eval dataset. The optimal threshold value that yields the highest number of correct cases is
    determined.

    Args:
      results: pandas Dataframe containing the eval results.
      result_dir: string location of the visualized images.
      lower: (optional) float defaults to 0.01, start of the search range.
      upper: (optional) float defaults to 0.5, end of the search range.
      step: (optional) float defeaults to 0.01, step of the search range.

    Returns:
      the optimal threshold value of the model.
    """
    thresholds = np.arange(lower, upper, step)
    res = []
    for threshold in thresholds:
        correct = 0
        total = 0
        for true, pred in zip(results[LABEL_NAME], results[DT_NAME]):
            if float(pred) > threshold and int(true) == 1:
                correct += 1
            elif float(pred) < threshold and int(true) == 0:
                correct += 1
            total += 1
        res.append(correct / total) 
    
    fig, ax = plt.subplots()
    ax.plot(thresholds, res, color='purple')
    ax.set_title('Correct vs. Threshold')
    ax.set_ylabel('correct / total')
    ax.set_xlabel('thresholds')

    plot_path = os.path.join(result_dir, "correct_thresholds_curve.png")
    plt.savefig(plot_path)
    print(f"plot saved at {plot_path}")

    optimal_th = thresholds[res.index(max(res))]
    return optimal_th


def precision_recall_curve(results, result_dir, threshold_start=0.01, threshold_end=0.5, step=0.01):
    """Plot and save the precision recall curve using the data in the results table.

    Args:
      results: pandas Dataframe containing the eval results.
      result_dir: string location of the visualized images.
      threshold_start: (optional) float defaults to 0.01, start of the threshold range.
      threshold_end: (optional) float defaults to 0.5, end of the threshold range.
      step: (optional) float defeaults to 0.01, step of the threshold range.
    """
    thresholds = np.arange(threshold_start, threshold_end, step)
    precision_arr = [0] * len(thresholds)
    recall_arr = [0] * len(thresholds)

    for i, threshold in enumerate(thresholds):
        tp = fp = tn = fn = 0
        for true, pred in zip(results[LABEL_NAME], results[DT_NAME]):
            if float(pred) > threshold and int(true) == 1:
                tp += 1
            elif float(pred) > threshold and int(true) == 0:
                fp += 1
            elif float(pred) < threshold and int(true) == 0:
                tn += 1
            else:
                fn += 1
        
        if tp + fp != 0:
            precision_arr[i] = tp / (tp + fp)

        if tp + fn != 0:
            recall_arr[i] = tp / (tp + fn)
    
    #fig, ax = plt.subplots()
    disp = PrecisionRecallDisplay(precision=precision_arr, recall=recall_arr)
    disp.plot(color="darkorange")
    _ = disp.ax_.set_title(f"Precision-Recall curve (Positive Label: 1; Threshold range {thresholds[0]} : {thresholds[-1]})")

    plot_path = os.path.join(result_dir, "precision_recall_curve.png")
    plt.savefig(plot_path)
    print(f"plot saved at {plot_path}")


def region_img_show(variant, anchor_data, sim_data, result_dir):
    """Visualize images using the anchor and simulation embedding data. 

    Args:
      variant: pandas dataframe describing the information of the variant.
      anchor_data: numpy array storing the anchor embeddings.
      anchor_data: numpy array storing the simulation embeddings.
      result_dir: string location of the visualized images.
    """
    fig = plt.figure(figsize=(25, 8))
    
    channels = anchor_data.shape[-1]
    rows, cols = 2, channels

    for c in range(channels):
        fig.add_subplot(rows, cols, c + 1)
        plt.imshow(anchor_data[:,:,c])
        plt.axis("off")
        plt.title(f"anchor dim {c + 1}")

        fig.add_subplot(rows, cols, cols + (c + 1))
        plt.imshow(sim_data[:,:,c])
        plt.axis("off")
        plt.title(f"sim dim {c + 1}")

    variant_info_str = f"contig: {variant[CONTIG_IDX]}, start: {variant[START_IDX]}, len: {variant[LEN_IDX]}, genotype: {variant[LABEL_IDX]}"
    fig.suptitle(variant_info_str, x=0.08)
    fig.tight_layout()
    image_name_str = f"{variant[CONTIG_IDX]}_{variant[START_IDX]}_{variant[LEN_IDX]}.png"
    plot_path = os.path.join(result_dir, image_name_str)
    plt.savefig(plot_path)
    plt.close()


def sample_region_img(results, result_data_dir, result_dir):
    """Visualize images used for training. Images are organized into 4 categories
    based on model's prediction on them: true positives, true negatives, false
    positives, false negatives. 

    Args:
      results: pandas Dataframe containing the eval results.
      result_data_dir: string location of the variant data.
      result_dir: string location of the visualized images.
    """
    anchor_path = os.path.join(result_data_dir, "anchor.npy")
    sim_path = os.path.join(result_data_dir, "sim.npy")
    anchor_data = np.load(anchor_path, allow_pickle=True)
    sim_data = np.load(sim_path, allow_pickle=True)

    results = results[:5].transpose()
    threshold = 0.8
    fp, fn, tp, tn = [], [], [], []
    for i, variant in enumerate(results):
        genotype = int(variant[LABEL_IDX])
        pred_genotype = 1 if float(variant[DISTANCE_IDX]) >= threshold else 0
        # find pred hom.ref, true non-hom.ref
        if pred_genotype == 0 and genotype == 1:
            false_negative_dir = os.path.join(result_dir, "fp")
            region_img_show(variant, anchor_data[i], sim_data[i], false_negative_dir)
        if pred_genotype == 1 and genotype == 0:
            false_positive_dir = os.path.join(result_dir, "fn")
            region_img_show(variant, anchor_data[i], sim_data[i], false_positive_dir)
        if pred_genotype == 0 and genotype == 0:
            false_negative_dir = os.path.join(result_dir, "tp")
            region_img_show(variant, anchor_data[i], sim_data[i], false_negative_dir)
        if pred_genotype == 1 and genotype == 1:
            false_positive_dir = os.path.join(result_dir, "tn")
            region_img_show(variant, anchor_data[i], sim_data[i], false_positive_dir)
        if i > 20: break # save only the first 20 images


def concordance_analysis(results: pd.DataFrame, refine_data_path: str, result_data_dir: str, th: float):
    """Perform the "refine" step on the specified test data. The method will print out the accuracy after refinement,
    and the corresponding refine dataset with appended refined genotype. 

    Args:
      results: pandas Dataframe containing the eval results.
      result_data_path: string location of the test file.
      result_data_dir: string location of the variant data.
      th: float. the optimal model threshold.
    """
    original_df = pd.read_csv(refine_data_path, sep='\t')
    results_df = results[[ID_NAME, DT_NAME]]
    results_df.rename(columns={ID_NAME:"ID", DT_NAME:"PDT"}, inplace=True)
    with_pred = original_df.merge(results_df, on="ID")
    with_pred["PDT"] = pd.to_numeric(with_pred["PDT"])
    with_pred["Refine"] = np.where(with_pred["PDT"] < th, True, False)
    
    def categorise(row):
        if row["Refine"] and row["OGT"] != ".":
            return row["OGT"]
        else:
            return row["GT"]

    with_pred["Final"] = with_pred.apply(lambda row: categorise(row), axis=1)
    correct = with_pred[with_pred["Final"] == with_pred["GIAB_GT"]]

    output = (
        f"optimal threshold: {th:>20.2f}\n"
        f"correct predictions: {len(correct.index):>18}\n"
        f"refined accuracy: {len(correct.index) / len(with_pred.index):>21.4f}\n"
    )

    print(output)
    result_data_path = os.path.join(result_data_dir, "refined_test5.tsv")
    with_pred.to_csv(result_data_path, sep='\t', index=False)
    print(f"refined results saved at {result_data_path}")


def refine_results(cfg):
    """Final step of the refine pipeline. Save results after running the model as
    a refiner on test dataset. 

    Args:
        cfg: dictionary contains configuration information. 
    """
    os.makedirs(cfg.mode.result_output_dir, exist_ok=True)

    result_path = os.path.join(cfg.mode.result_data_dir, "pred_results.csv")
    results = pd.read_csv(result_path)
    
    if cfg.mode.save_embedding:
        plot_pca_fig(results, cfg.mode.result_data_dir, cfg.mode.result_output_dir)
    if cfg.mode.save_image:
        sample_region_img(results, cfg.mode.result_data_dir, cfg.mode.result_output_dir)

    # precision recall curve
    precision_recall_curve(results, cfg.mode.result_output_dir)

    # visualize thresholds
    th = th_search(results, cfg.mode.result_output_dir)
    concordance_analysis(results, cfg.mode.refine_data_path, cfg.mode.result_output_dir, th)