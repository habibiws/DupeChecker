import numpy as np
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity
import os
from PIL import Image

class DuplicateDetector:
    def __init__(self, model_path, config):
        self.config = config
        self.model = tf.keras.models.load_model(
            model_path, 
            custom_objects={'contrastive_loss': self.contrastive_loss}
        )
        self.embeddings_cache = {}
        
    def contrastive_loss(self, y_true, y_pred, margin=1.0):
        """Contrastive loss function (diperlukan untuk loading model)"""
        y_true = tf.cast(y_true, tf.float32)
        square_pred = tf.square(y_pred)
        margin_square = tf.square(tf.maximum(margin - y_pred, 0))
        return tf.reduce_mean(y_true * square_pred + (1 - y_true) * margin_square)
    
    def preprocess_image(self, image_path):
        """Preprocess gambar untuk inference"""
        image = tf.io.read_file(image_path)
        image = tf.image.decode_image(image, channels=3)
        image = tf.image.resize(image, self.config.IMAGE_SIZE)
        image = tf.cast(image, tf.float32) / 255.0
        image = tf.expand_dims(image, 0)  # Add batch dimension
        return image
    
    def get_image_embedding(self, image_path):
        """Mendapatkan embedding dari gambar"""
        if image_path in self.embeddings_cache:
            return self.embeddings_cache[image_path]
            
        # Ambil base network dari Siamese model dengan namanya
        base_network = self.model.get_layer("embedding_network")  # Sesuaikan index jika perlu
        
        image = self.preprocess_image(image_path)
        embedding = base_network.predict(image, verbose=0)
        
        # Cache embedding
        self.embeddings_cache[image_path] = embedding
        return embedding
    
    def calculate_similarity(self, image1_path, image2_path):
        """Menghitung similarity antara dua gambar"""
        embedding1 = self.get_image_embedding(image1_path)
        embedding2 = self.get_image_embedding(image2_path)
        
        # Menggunakan cosine similarity
        similarity = cosine_similarity(embedding1, embedding2)[0][0]
        return similarity
    
    def find_duplicates_in_folder(self, folder_path):
        """Mencari duplikasi dalam folder"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = []
        
        # Kumpulkan semua file gambar
        for filename in os.listdir(folder_path):
            if any(filename.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(folder_path, filename))
        
        duplicates = []
        processed_pairs = set()
        
        print(f"Memproses {len(image_files)} gambar...")
        
        # Bandingkan setiap pasangan gambar
        for i, img1 in enumerate(image_files):
            for j, img2 in enumerate(image_files[i+1:], i+1):
                pair = tuple(sorted([img1, img2]))
                if pair in processed_pairs:
                    continue
                    
                processed_pairs.add(pair)
                similarity = self.calculate_similarity(img1, img2)
                
                if similarity > self.config.SIMILARITY_THRESHOLD:
                    duplicates.append({
                        'image1': img1,
                        'image2': img2,
                        'similarity': similarity
                    })
                    print(f"Duplikasi ditemukan: {os.path.basename(img1)} <-> {os.path.basename(img2)} (similarity: {similarity:.3f})")
        
        return duplicates
    
    def find_similar_images(self, query_image_path, search_folder):
        """Mencari gambar yang mirip dengan query image"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        similar_images = []
        
        # Dapatkan embedding query image
        query_embedding = self.get_image_embedding(query_image_path)
        
        # Cari di folder
        for filename in os.listdir(search_folder):
            if any(filename.lower().endswith(ext) for ext in image_extensions):
                image_path = os.path.join(search_folder, filename)
                if image_path == query_image_path:
                    continue
                    
                similarity = self.calculate_similarity(query_image_path, image_path)
                
                if similarity > self.config.SIMILARITY_THRESHOLD:
                    similar_images.append({
                        'image_path': image_path,
                        'similarity': similarity
                    })
        
        # Sort berdasarkan similarity
        similar_images.sort(key=lambda x: x['similarity'], reverse=True)
        return similar_images
