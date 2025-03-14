import os
import numpy as np

def read_embedding_file(embedding_file_path):
    """
    Read an embedding file (.npz) that was created by the embedding_images function
    and return the data in the same format.
    
    Args:
        embedding_file_path (str): Path to the .npz embedding file
        
    Returns:
        tuple: (image_classes, image_class_names, embeddings, embedding_file_path)
    """
    if not os.path.exists(embedding_file_path):
        print(f"Embedding file not found: {embedding_file_path}")
        return [], [], None, embedding_file_path
    
    try:
        # Load the npz file
        data = np.load(embedding_file_path)
        # Print all keys inside the file
        print("Keys in the .npz file:", data.files)

        # Access a specific array (replace 'key_name' with the actual key)
        key_name = data.files[0]  # Assuming the first key
        embeddings = data[key_name]

        # Print the shape of the extracted embeddings
        print(f"Shape of '{key_name}' embeddings:", embeddings.shape)

        # If you want to iterate over all stored arrays
        for key in data.files:
            print(f"Key: {key}, Shape: {data[key].shape}")

        # Extract the required components
        image_classes = data["image_classes"]
        image_class_names = data["image_class_names"]
        embeddings = data["embeddings"]
        
        print(">>>> image_classes info:")
        print(image_classes.shape)
        print(image_class_names)
        
        return image_classes, image_class_names, embeddings, embedding_file_path
        
    except Exception as e:
        print(f"Error reading embedding file: {e}")
        return [], [], None, embedding_file_path


# model_file = "models/GhostFaceNet_W1.3_S2_ArcFace.h5"
# # embeddings_file = "known_user/embeddings.npz"
# embeddings_file = "known_user/image-dataset_aligned_112_112_embedding.npz"
# image_path = 'check/ibe1.jpg'
# dist_thresh = 0.6
# # known_user = '/kaggle/working/image-dataset_aligned_112_112/'
# known_user = "known_user/embeddings.npz"
# # det, face_model = init_det_and_emb_model(model_file)
# # image_classes, image_class_name, embeddings = embedding_images(det, face_model, known_user)
# image_classes, image_class_name, embeddings, embbeddings_file_path = read_embedding_file(embeddings_file)
# image_classes, image_class_name, embeddings, embbeddings_file_path = read_embedding_file(embeddings_file)
# print(image_class_name)
# frame = imread(image_path)
# # rec_dist, rec_class, bbs = recognize_image(det, face_model, image_classes, embeddings, frame, dist_thresh)
# rec_dist, rec_class, bbs, ccs = recognize_image(det, face_model, image_classes, image_class_name, embeddings, frame, dist_thresh)
# print(rec_class)
# print(rec_dist)



def read_embedding_file_new(embedding_file_path):
    if not os.path.exists(embedding_file_path):
        print(f"Embedding file not found: {embedding_file_path}")
        return [], [], None, embedding_file_path
    
    try:
        data = np.load(embedding_file_path)
        print("Keys in the .npz file:", data.files)
        key_name = data.files[0]
        embeddings = data[key_name]
        print(f"Shape of '{key_name}' embeddings:", embeddings.shape)
        for key in data.files:
            print(f"Key: {key}, Shape: {data[key].shape}")
        image_classes = data["imm_classes"]
        image_class_names = data["imm_class_names"]
        embeddings = data["embs"]
        
        print(">>>> image_classes info:")
        print(image_classes.shape)
        
        return image_classes, image_class_names, embeddings, embedding_file_path
        
    except Exception as e:
        print(f"Error reading embedding file: {e}")
        return [], [], None, embedding_file_path
    
embeddings_file = "known_user/embeddings_new.npz"
image_classes, image_class_name, embeddings, embbeddings_file_path = read_embedding_file_new(embeddings_file)
print(image_class_name)
# pip install tensorflow==2.8.0
# pip install keras==2.8.0
# pip install keras_cv_attention_models
# pip install glob2