import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

def create_directory_structure(base_path, outer_names, inner_names):
    for outer_name in outer_names:
        for inner_name in inner_names:
            os.makedirs(os.path.join(base_path, outer_name, inner_name), exist_ok=True)

def process_and_save_image(row, base_path, category, emotion_map):
    try:
        emotion = emotion_map[row['emotion']]
        pixels = np.fromstring(row['pixels'], dtype=np.uint8, sep=' ').reshape(48, 48)
        img = Image.fromarray(pixels)
        filename = os.path.join(base_path, category, emotion, f'im{row["file_id"]}.png')
        img.save(filename)
        return emotion, category
    except Exception as e:
        print(f"Error processing image {row['file_id']}: {e}")
        return None, None

def process_dataset(df, base_path, emotion_map, n_workers=None):
    if n_workers is None:
        n_workers = os.cpu_count() or 4  # Default to 4 if os.cpu_count() is None

    counters = {category: {emotion: 0 for emotion in emotion_map.values()} for category in ['train', 'test']}
    
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        future_to_row = {executor.submit(process_and_save_image, row, base_path, 
                                         'train' if i < 28709 else 'test', emotion_map): i 
                         for i, row in df.iterrows()}
        
        for future in tqdm(as_completed(future_to_row), total=len(df), desc="Processing images"):
            emotion, category = future.result()
            if emotion and category:
                counters[category][emotion] += 1
    
    return counters

def main():
    base_path = 'data'
    outer_names = ['test', 'train']
    emotion_map = {0: 'angry', 1: 'disgusted', 2: 'fearful', 3: 'happy', 
                   4: 'sad', 5: 'surprised', 6: 'neutral'}

    create_directory_structure(base_path, outer_names, emotion_map.values())

    df = pd.read_csv('./fer2013.csv')
    df['file_id'] = df.groupby('emotion').cumcount()

    print("Processing and saving images...")
    counters = process_dataset(df, base_path, emotion_map)

    print("\nDataset summary:")
    for category in outer_names:
        print(f"\n{category.capitalize()} set:")
        for emotion, count in counters[category].items():
            print(f"  {emotion.capitalize()}: {count}")

    total_images = sum(sum(counters[cat].values()) for cat in outer_names)
    print(f"\nTotal images processed: {total_images}")

if __name__ == "__main__":
    main()
