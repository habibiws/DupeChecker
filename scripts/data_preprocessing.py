import tensorflow as tf
import numpy as np
import os
from PIL import Image
import random

class DataPreprocessor:
    def __init__(self, config):
        self.config = config
        
    def load_and_preprocess_image(self, image_path):
        """Load dan preprocess gambar dengan penanganan error"""
        try:
            image = tf.io.read_file(image_path)
            # Gunakan decode_jpeg atau decode_png jika Anda tahu formatnya, 
            # jika tidak decode_image lebih fleksibel.
            image = tf.image.decode_image(image, channels=3, expand_animations=False)
        
            # PENTING: Set shape secara manual setelah decode
            # Ini adalah perbaikan utama untuk error 'contains no shape'
            image.set_shape([None, None, 3])

            image = tf.image.resize(image, self.config.IMAGE_SIZE)
            image = tf.cast(image, tf.float32) / 255.0
            return image
        except Exception as e:
            # Jika terjadi error saat memproses satu gambar, cetak peringatan
            # dan kembalikan gambar kosong (placeholder) agar proses tidak berhenti total.
            tf.print(f"Error processing image {image_path}: {e}")
            # Kembalikan tensor hitam dengan shape yang benar
            return tf.zeros((*self.config.IMAGE_SIZE, 3))
    
    def create_pairs(self, image_paths, labels):
        """Membuat pairs untuk Siamese Network"""
        pairs = []
        pair_labels = []
        
        # Buat positive pairs (gambar yang sama/mirip)
        for i in range(len(image_paths)):
            for j in range(i + 1, len(image_paths)):
                if labels[i] == labels[j]:  # Same class = positive pair
                    pairs.append([image_paths[i], image_paths[j]])
                    pair_labels.append(1)
        
        # Buat negative pairs (gambar yang berbeda)
        for i in range(len(image_paths)):
            for j in range(len(image_paths)):
                if labels[i] != labels[j]:  # Different class = negative pair
                    pairs.append([image_paths[i], image_paths[j]])
                    pair_labels.append(0)
        
        # Shuffle pairs
        combined = list(zip(pairs, pair_labels))
        random.shuffle(combined)
        pairs, pair_labels = zip(*combined)
        
        return list(pairs), list(pair_labels)
    
    def create_dataset(self, pairs, labels):
        """Membuat TensorFlow dataset"""
        def load_pair(pair, label):
            img1 = self.load_and_preprocess_image(pair[0])
            img2 = self.load_and_preprocess_image(pair[1])
            return (img1, img2), label
        
        dataset = tf.data.Dataset.from_tensor_slices((pairs, labels))
        dataset = dataset.map(load_pair, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(self.config.BATCH_SIZE)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
