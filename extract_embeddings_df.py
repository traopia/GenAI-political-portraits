import os
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
import open_clip

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def create_image_dataframe(base_dir='pictures'):
    data = []


    for country in os.listdir(base_dir) if base_dir else ["China", "Italy", "Netherlands"]:
        country_path = os.path.join(base_dir, country) if base_dir else country
        if not os.path.isdir(country_path):
            continue

        for prompt_type in os.listdir(country_path):
            prompt_path = os.path.join(country_path, prompt_type)
            if not os.path.isdir(prompt_path):
                continue

            # Normalize prompt name: 'Prompt - China' -> 'China' or 'English'
            if 'English' in prompt_type:
                prompt = 'English'
            else:
                prompt = country  # The prompt is in the country's language

            for model in os.listdir(prompt_path):
                model_path = os.path.join(prompt_path, model)
                if not os.path.isdir(model_path):
                    continue

                for img_file in os.listdir(model_path):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                        image_path = os.path.join(model_path, img_file)
                        data.append({
                            'country': country,
                            'prompt': prompt,
                            'model': model,
                            'image_path': image_path
                        })

    df = pd.DataFrame(data)
    return df





def add_clip_embeddings(df, device):
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
    model = model.to(device)
    model.eval()

    clip_embeddings = []

    for path in df['image_path']:
        try:
            image = preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
            with torch.no_grad():
                embedding = model.encode_image(image).squeeze().cpu().numpy()
        except Exception as e:
            print(f"Error processing {path}: {e}")
            embedding = None
        clip_embeddings.append(embedding)

    df['clip'] = clip_embeddings
    return df


def main():
    df = create_image_dataframe("")
    df = add_clip_embeddings(df, device)
    df.to_json("data_genai.json",orient = "records" , lines = True)


if __name__ == "__main__":
    main()