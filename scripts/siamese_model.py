import tensorflow as tf
from tensorflow import keras
from keras import layers, Model
from keras.applications import ResNet50

class SiameseNetwork:
    def __init__(self, config):
        self.config = config
        self.model = None
        
    def create_base_network(self):
        """Membuat base network untuk feature extraction"""
        # Menggunakan ResNet50 sebagai backbone (bisa diganti sesuai kebutuhan)
        base_model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=(*self.config.IMAGE_SIZE, 3)
        )
        
        # Freeze beberapa layer pertama (optional)
        for layer in base_model.layers[:-10]:
            layer.trainable = False
            
        # Tambahkan layer untuk embedding
        x = base_model.output
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(self.config.EMBEDDING_SIZE, activation='linear')(x)
        x = layers.Lambda(lambda x: tf.nn.l2_normalize(x, axis=1))(x)
        
        model = Model(inputs=base_model.input, outputs=x)
        return model
    
    def euclidean_distance(self, vectors):
        """Menghitung Euclidean distance antara dua vector"""
        x, y = vectors
        sum_square = tf.reduce_sum(tf.square(x - y), axis=1, keepdims=True)
        return tf.sqrt(tf.maximum(sum_square, tf.keras.backend.epsilon()))
    
    def contrastive_loss(self, y_true, y_pred, margin=1.0):
        """Contrastive loss function untuk Siamese Network"""
        y_true = tf.cast(y_true, tf.float32)
        square_pred = tf.square(y_pred)
        margin_square = tf.square(tf.maximum(margin - y_pred, 0))
        return tf.reduce_mean(y_true * square_pred + (1 - y_true) * margin_square)
    
    def build_siamese_model(self):
        """Membangun Siamese Network"""
        # Input untuk dua gambar
        input_a = layers.Input(shape=(*self.config.IMAGE_SIZE, 3))
        input_b = layers.Input(shape=(*self.config.IMAGE_SIZE, 3))
        
        # Base network yang akan di-share
        base_network = self.create_base_network()
        base_network._name = "embedding_network"

        # Proses kedua input dengan network yang sama
        processed_a = base_network(input_a)
        processed_b = base_network(input_b)
        
        # Hitung distance
        distance = layers.Lambda(self.euclidean_distance)([processed_a, processed_b])
        
        # Buat model
        self.model = Model(inputs=[input_a, input_b], outputs=distance)
        
        # Compile model
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.LEARNING_RATE),
            loss=self.contrastive_loss,
            metrics=['accuracy']
        )
        
        return self.model
    
    def get_embedding(self, image):
        """Mendapatkan embedding dari satu gambar"""
        if self.model is None:
            raise ValueError("Model belum dibangun. Panggil build_siamese_model() terlebih dahulu.")
        
        # Ambil base network dari model Siamese
        base_network = self.model.layers[2]  # Sesuaikan index jika perlu
        return base_network.predict(image)
