# Configuration file - Isi variabel-variabel ini sesuai kebutuhan Anda
class Config:
    # Dataset Configuration  
    # Path ke dataset Anda
    DATASET_PATH = "./AirbnbData"
    IMAGE_SIZE = (224, 224)  # Ukuran gambar yang akan diproses
    BATCH_SIZE = 32  # Batch size untuk training
    
    # Model Configuration
    EMBEDDING_SIZE = 128  # Ukuran embedding vector
    LEARNING_RATE = 0.001  # Learning rate
    EPOCHS = 50  # Jumlah epoch training
    
    # Similarity Threshold
    SIMILARITY_THRESHOLD = 0.5  # Threshold untuk menentukan duplikasi (0-1)
    
    # Model Save Path
    MODEL_SAVE_PATH = "models/siamese_model.h5"
    
    # Data Split
    TRAIN_SPLIT = 0.8
    VAL_SPLIT = 0.2
