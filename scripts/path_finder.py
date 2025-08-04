import os
import re

def load_data_from_filenames(root_dir):
    """
    Memindai direktori dataset dan secara otomatis membuat daftar image_paths dan labels
    berdasarkan ID unik yang ada di nama file.

    Contoh nama file: seattle_3269390_1.jpg -> ID uniknya adalah '3269390'

    Args:
        root_dir (str): Path ke direktori utama dataset.

    Returns:
        tuple: Sebuah tuple berisi dua list (image_paths, labels).
    """
    image_paths = []
    labels = []
    
    # Memetakan dari ID unik (string) ke label (integer)
    class_id_to_idx = {}
    
    print("Memindai file dan mengekstrak label dari nama file...")

    # Menggunakan os.walk untuk memindai semua folder dan sub-folder
    for subdir, dirs, files in os.walk(root_dir):
        for filename in files:
            # Memastikan file adalah gambar
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                try:
                    # Ekstrak ID unik dari nama file.
                    # Asumsi: ID adalah bagian sebelum underscore terakhir.
                    # Contoh: 'seattle_3269390_1' -> 'seattle_3269390'
                    base_name = os.path.splitext(filename)[0] # Hapus ekstensi .jpg
                    parts = base_name.split('_')
                    class_id = '_'.join(parts[:-1]) # Gabungkan semua kecuali bagian terakhir

                    # Jika Anda yakin ID-nya selalu angka di tengah:
                    # class_id = parts[1] # Ambil '3269390'

                    # Jika class_id belum pernah ditemukan, berikan label integer baru
                    if class_id not in class_id_to_idx:
                        class_id_to_idx[class_id] = len(class_id_to_idx)
                    
                    label_idx = class_id_to_idx[class_id]
                    
                    # Tambahkan path gambar dan labelnya ke daftar
                    image_path = os.path.join(subdir, filename)
                    image_paths.append(image_path)
                    labels.append(label_idx)

                except IndexError:
                    print(f"Peringatan: Nama file '{filename}' tidak sesuai format yang diharapkan dan akan dilewati.")

    print(f"Total kelas unik ditemukan: {len(class_id_to_idx)}")
    return image_paths, labels