# main.py

import os
from config import Config
from data_preprocessing import DataPreprocessor
from siamese_model import SiameseNetwork
from trainer import SiameseTrainer
from duplicate_detector import DuplicateDetector

# ===================================================================
# == LETAKKAN DEFINISI FUNGSI BARU DI SINI ==
# ===================================================================
def load_data_from_filenames(root_dir):
    """
    Memindai direktori dataset dan secara otomatis membuat daftar image_paths dan labels
    berdasarkan ID unik yang ada di nama file.
    """
    image_paths = []
    labels = []
    class_id_to_idx = {}
    
    print("Memindai file dan mengekstrak label dari nama file...")

    for subdir, dirs, files in os.walk(root_dir):
        for filename in files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                try:
                    base_name = os.path.splitext(filename)[0]
                    parts = base_name.split('_')
                    class_id = '_'.join(parts[:-1])

                    if class_id not in class_id_to_idx:
                        class_id_to_idx[class_id] = len(class_id_to_idx)
                    
                    label_idx = class_id_to_idx[class_id]
                    
                    image_path = os.path.join(subdir, filename)
                    image_paths.append(image_path)
                    labels.append(label_idx)

                except IndexError:
                    print(f"Peringatan: Nama file '{filename}' tidak sesuai format dan akan dilewati.")

    print(f"Total kelas unik ditemukan: {len(class_id_to_idx)}")
    return image_paths, labels
# ===================================================================

def train_model():
    """Function untuk training model"""
    config = Config()
    
    # Panggil fungsi untuk memuat data secara otomatis
    print("Memuat data dari folder...")
    image_paths, labels = load_data_from_filenames(config.DATASET_PATH)
    print(f"Total gambar ditemukan: {len(image_paths)}")
    print(f"Total label ditemukan: {len(labels)}")

    if not image_paths:
        print("Tidak ada gambar yang ditemukan. Periksa DATASET_PATH di config.py.")
        return
    
    # Preprocessing data
    preprocessor = DataPreprocessor(config)
    pairs, pair_labels = preprocessor.create_pairs(image_paths, labels)
    
    # Split data
    split_idx = int(len(pairs) * config.TRAIN_SPLIT)
    train_pairs = pairs[:split_idx]
    train_labels = pair_labels[:split_idx]
    val_pairs = pairs[split_idx:]
    val_labels = pair_labels[split_idx:]
    
    # Buat dataset
    train_dataset = preprocessor.create_dataset(train_pairs, train_labels)
    val_dataset = preprocessor.create_dataset(val_pairs, val_labels)
    
    # Buat dan training model
    siamese_net = SiameseNetwork(config)
    model = siamese_net.build_siamese_model()
    
    trainer = SiameseTrainer(model, config)
    history = trainer.train(train_dataset, val_dataset)
    
    # Plot hasil training
    trainer.plot_training_history()
    
    print(f"Model berhasil disimpan di: {config.MODEL_SAVE_PATH}")

def detect_duplicates():
    """Function untuk deteksi duplikasi"""
    config = Config()
    
    # Load trained model
    detector = DuplicateDetector(config.MODEL_SAVE_PATH, config)
    
    # TODO: Ganti dengan path folder yang ingin dicek
    folder_path = "./AirbnbData/Test Data"  # Ganti dengan path folder yang sesuai
    
    # Cari duplikasi
    duplicates = detector.find_duplicates_in_folder(folder_path)
    
    print(f"\nDitemukan {len(duplicates)} pasangan gambar duplikat:")
    for dup in duplicates:
        print(f"- {os.path.basename(dup['image1'])} <-> {os.path.basename(dup['image2'])} (similarity: {dup['similarity']:.3f})")

def find_similar():
    """Function untuk mencari gambar yang mirip"""
    config = Config()
    
    # Load trained model
    detector = DuplicateDetector(config.MODEL_SAVE_PATH, config)
    
    # TODO: Ganti dengan path gambar query dan folder pencarian
    query_image = "path/to/query/image.jpg"
    search_folder = "path/to/search/folder"
    
    # Cari gambar mirip
    similar_images = detector.find_similar_images(query_image, search_folder)
    
    print(f"\nDitemukan {len(similar_images)} gambar yang mirip:")
    for img in similar_images:
        print(f"- {os.path.basename(img['image_path'])} (similarity: {img['similarity']:.3f})")

if __name__ == "__main__":
    # Pilih mode operasi
    mode = "train"  # Ganti dengan: "train", "detect", atau "similar"
    
    if mode == "train":
        train_model()
    elif mode == "detect":
        detect_duplicates()
    elif mode == "similar":
        find_similar()
    else:
        print("Mode tidak valid. Pilih: 'train', 'detect', atau 'similar'")