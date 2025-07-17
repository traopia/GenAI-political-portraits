import os 
from extract_embeddings_df import create_image_dataframe, add_clip_embeddings

import torch 
import pandas as pd
import numpy as np

import numpy as np
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import open_clip


from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap


import matplotlib.pyplot as plt
import seaborn as sns


from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.patches as mpatches


device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"





def compute_country_predictions(df, device=device):
    # Load CLIP model
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    model = model.to(device)
    model.eval()

    # Get unique country names
    countries = sorted(df['country'].unique())
    #countries.append("English")
    country_texts = [f"{c}" for c in countries]  # Add context if you like

    # Encode country names as text embeddings
    with torch.no_grad():
        tokenized = tokenizer(country_texts).to(device)

        text_features = model.encode_text(tokenized)
        text_features = F.normalize(text_features, dim=-1)  # L2-normalize

    # Convert image embeddings to tensor and normalize
    image_features = torch.tensor(np.stack(df['clip'].values)).to(torch.float32).to(device)

    image_features = F.normalize(image_features, dim=-1)

    # Compute cosine similarity (dot product after normalization)
    similarities = image_features @ text_features.T  # shape: [num_images, num_countries]

    # Apply softmax to get probabilities
    probs = F.softmax(similarities, dim=1).cpu().numpy()

    # Get predictions and top match
    predicted_idxs = np.argmax(probs, axis=1)
    predicted_countries = [countries[i] for i in predicted_idxs]

    # Add to dataframe
    df['predicted_country'] = predicted_countries
    df['country_correct'] = df['predicted_country'] == df['country']

    return df, probs





def compute_politician_prediction(df, device=device):
    # Load CLIP model
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    model = model.to(device)
    model.eval()

    # Define class texts
    class_names = ["Photo of a politician", "Photo"]
    
    # Encode the class texts
    with torch.no_grad():
        text_tokens = tokenizer(class_names).to(device)

        text_features = model.encode_text(text_tokens)
        text_features = F.normalize(text_features, dim=-1)  # shape: [2, d]

    # Prepare image features
    image_features = torch.tensor(np.stack(df['clip'].values)).to(torch.float32).to(device)

    image_features = F.normalize(image_features, dim=-1)

    # Compute similarity scores (dot product)
    logits = image_features @ text_features.T  # shape: [N_images, 2]

    # Apply softmax across classes
    probs = F.softmax(logits, dim=-1).cpu().numpy()

    # Get predictions
    pred_indices = np.argmax(probs, axis=1)
    predicted_labels = [class_names[i] for i in pred_indices]

    # Store results
    # df['politician_prob'] = probs[:, 0]
    # df['other_prob'] = probs[:, 1]
    df['predicted_label'] = predicted_labels
    df['politician'] = df['predicted_label'] == "Photo of a politician"

    return df


def compute_accuracy_tables(df):
    acc_by_country = df.groupby('country')[['country_correct', 'politician']].mean().reset_index()
    acc_by_model = df.groupby('model')[['country_correct', 'politician']].mean().reset_index()
    acc_by_prompt = df.groupby('prompt')[['country_correct', 'politician']].mean().reset_index()
    acc_by_all = df.groupby(["combo_label"])[['country_correct', 'politician']].mean().reset_index()
    
    return acc_by_country, acc_by_model, acc_by_prompt, acc_by_all



def plot_association_scores(acc_by_all):
# Set style and palette
    sns.set(style="whitegrid")
    custom_palette = sns.color_palette("Set2")
    acc_by_all = acc_by_all.rename(columns={"country_correct": "nationality association", "politician": "politician association"})
    # Sort the data
    #acc_sorted = acc_by_all.sort_values(by="country accuracy")

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(14, 6))

    # Plot with correct alignment
    acc_by_all.plot(
        kind="bar",
        x="combo_label",
        ax=ax,
        color=custom_palette,
        width=0.8
    )

    # Fix label alignment
    ax.set_xticks(range(len(acc_by_all)))
    ax.set_xticklabels(acc_by_all["combo_label"], rotation=60, ha="right")

    # Axis labels and title
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_xlabel(" Country | Language | Model ", fontsize=12)
    ax.set_title("Association scores", fontsize=14)

    # Legend outside
    ax.legend(title="Metric", bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=10)

    plt.tight_layout()
    plt.savefig("results/association_scores.png")
    plt.show()
    


def reduce_embeddings(df, method='umap'):
    image_embs = np.stack(df['clip'].values)
    reducer = {
        'umap': umap.UMAP(metric="cosine", random_state=5),
        'tsne': TSNE(n_components=2, random_state=42),
        'pca': PCA(n_components=2)
    }[method]

    reduced = reducer.fit_transform(image_embs)
    df['x'] = reduced[:, 0]
    df['y'] = reduced[:, 1]
    return df



def plot_embeddings_with_double_borders(df, figsize=(20, 15), dot_size=100):
    fig, ax = plt.subplots(figsize=figsize)

    for _, row in df.iterrows():
        x, y = row['x'], row['y']
        img_path = row['image_path']
        country_correct = row.get('country_correct', True)
        politician_correct = row.get('politician', True)

        # Define colors
        inner_color = 'green' if country_correct else 'red'
        outer_color = 'blue' if politician_correct else 'orange'

        try:
            img = Image.open(img_path)
            img.thumbnail((50, 50))
            imagebox = OffsetImage(img, zoom=1)

            # Outer box (politician) - slightly bigger
            outer_ab = AnnotationBbox(
                imagebox, (x, y),
                bboxprops=dict(edgecolor=outer_color, linewidth=1.5),
                frameon=True
            )
            ax.add_artist(outer_ab)

            # Inner box (country) - on top, slightly smaller image
            inner_img = img.copy()
            inner_img.thumbnail((40, 40))
            inner_imagebox = OffsetImage(inner_img, zoom=1)
            inner_ab = AnnotationBbox(
                inner_imagebox, (x, y),
                bboxprops=dict(edgecolor=inner_color, linewidth=1.5),
                frameon=True
            )
            ax.add_artist(inner_ab)

        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            continue

    # Optional: base scatter just for layout reference
    ax.scatter(df['x'], df['y'],
               c=df['country'].astype('category').cat.codes,
               cmap='tab20', alpha=0.05, s=dot_size)

    # Legend
    legend_elements = [
        mpatches.Patch(color='green', label='Country correct'),
        mpatches.Patch(color='red', label='Country wrong'),
        mpatches.Patch(color='blue', label='Politician correct'),
        mpatches.Patch(color='orange', label='Politician wrong'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=12)

    ax.set_title('UMAP of CLIP Embedding Space – Highlighting Country & Politician Accuracy', fontsize=16)
    #ax.axis('off')
    plt.tight_layout()
    plt.show()
    plt.savefig("results/embeddings_umap.png")

def plot_clusters(df):

    # Visualize by country, prompt, model, or combo
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    for ax, col in zip(axes.flat, ["country", "prompt", "model", "combo_label"]):
        if col == "combo_label":
            sns.scatterplot(data=df, x="x", y="y", hue=col, ax=ax, palette="tab20", s=30)
        else:
            
            sns.scatterplot(data=df, x="x", y="y", hue=col, ax=ax, palette="tab10", s=30)
        ax.set_title(f"UMAP colored by {col}")
        ax.legend(loc="best", fontsize=8)

    plt.tight_layout()
    plt.show()
    plt.savefig("results/embeddings_umap_clusters.png")

def main(analysis_type = "association_scores"):
    if os.path.exists("data_genai.json"):
        df = pd.read_json("data_genai.json",orient = "records" , lines = True)
    else:
        df = create_image_dataframe("")
        df = add_clip_embeddings(df, device)
        df.to_json("data_genai.json",orient = "records" , lines = True)
    df["clip"] = df["clip"].apply(lambda x: np.array(x))

    df, probs = compute_country_predictions(df)
    df = compute_politician_prediction(df)
    df["combo_label"] = df["country"].str.replace("China", "CN").str.replace("Italy", "IT").str.replace("Netherlands", "NL") + " | " + df["prompt"].str.replace("China", "Chinese").str.replace("Italy", "Italian").str.replace("Netherlands", "Dutch") + " | " + df["model"].str.replace("OpenAI", "DALL-E")

    if analysis_type == "association_scores":
        acc_by_country, acc_by_model, acc_by_prompt, acc_by_all = compute_accuracy_tables(df)
        plot_association_scores(acc_by_all)
    elif analysis_type == "embeddings_umap":
        df = reduce_embeddings(df, method='umap')
        plot_embeddings_with_double_borders(df)
        plot_clusters(df)

if __name__ == "__main__":
    main("association_scores")
