from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from PIL import Image
import os
import cv2
import numpy as np
import tensorflow as tf
from skimage.io import imread
from sklearn.preprocessing import normalize
from face_detector import YoloV5FaceDetector
from tqdm import tqdm
import glob2
print("START")

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load models and embeddings before running the Flask app
model_file = "models/GhostFaceNet_W1.3_S2_ArcFace.h5"
embeddings_file = "known_user/image-dataset_aligned_112_112_embedding.npz"
known_user = "known_user/image-dataset_aligned_112_112_embedding.npz"
# known_user = "known_user/embeddings.npz"

def init_det_and_emb_model(model_file, embeddings_file):
    det = YoloV5FaceDetector()
    face_model = tf.keras.models.load_model(model_file, compile=False)
    data = np.load(embeddings_file)
    image_classes, image_class_name, embeddings = data["image_classes"], data["image_class_names"], data["embeddings"]
    return det, face_model, image_classes, image_class_name, embeddings

def face_align_landmarks_sk(img, landmarks, image_size=(112, 112)):
    from skimage import transform
    src = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)
    ret = []
    for landmark in landmarks:
        tform = transform.SimilarityTransform()
        tform.estimate(landmark, src)
        ret.append(transform.warp(img, tform.inverse, output_shape=image_size))
    return (np.array(ret) * 255).astype(np.uint8)

def do_detect_in_image(image, det, image_format="BGR"):
    print('detecting face from image')
    imm_BGR = image if image_format == "BGR" else image[:, :, ::-1]
    imm_RGB = image[:, :, ::-1] if image_format == "BGR" else image
    bboxes, pps, ccs = det.__call__(imm_BGR)
    # nimgs = face_align_landmarks_sk(imm_RGB, pps)

    nimgs = face_align_landmarks_sk(imm_RGB, pps)
    bbs, ccs = bboxes[:, :4].astype("int"), bboxes[:, -1]
    return bbs, ccs, nimgs

def embedding_images(det, face_model, known_user, batch_size=32, force_reload=False):
    while known_user.endswith("/"):
        known_user = known_user[:-1]
    dest_pickle = os.path.join(known_user, os.path.basename(known_user) + "_embedding.npz")

    if force_reload == False and os.path.exists(dest_pickle):
        aa = np.load(dest_pickle)
        print(aa)
        image_classes, image_class_names, embeddings = aa["image_classes"], aa["image_class_names"], aa["embeddings"]
    else:
        if not os.path.exists(known_user):
            print('path not exist :', known_user)
            return [], [], None
        # data_gen = ImageDataGenerator(preprocessing_function=lambda img: (img - 127.5) * 0.0078125)
        # img_gen = data_gen.flow_from_directory(known_user, target_size=(112, 112), batch_size=1, class_mode='binary')
        image_names = glob2.glob(os.path.join(known_user, "*/*.jpg"))

        """ Detct faces in images, keep only those have exactly one face. """
        nimgs, image_classes, image_class_names = [], [], []
        for image_name in tqdm(image_names, "Detect"):
            img = imread(image_name)
            nimg = do_detect_in_image(img, det, image_format="RGB")[-1]
            if nimg.shape[0] > 0:
                nimgs.append(nimg[0])
                image_classes.append(os.path.basename(os.path.dirname(image_name)))
                image_class_names.append(os.path.basename(os.path.dirname(image_name)))

        """ Extract embedding info from aligned face images """
        steps = int(np.ceil(len(image_classes) / batch_size))
        nimgs = (np.array(nimgs) - 127.5) * 0.0078125
        embeddings = [face_model(nimgs[ii * batch_size : (ii + 1) * batch_size]) for ii in tqdm(range(steps), "Embedding")]

        embeddings = normalize(np.concatenate(embeddings, axis=0))
        image_classes = np.array(image_classes)
        image_class_names = np.array(image_class_names)
        np.savez_compressed(dest_pickle, embeddings=embeddings, image_classes=image_classes, image_class_names=image_class_names)

    print(">>>> image_classes info:")
    print(image_classes.shape)
    return image_classes, image_class_names, embeddings, dest_pickle
    
def recognize_image(det, face_model, image_class_name, embeddings, frame, dist_thresh=0.6, image_format='BGR'):
    print('start recognize image')
    if isinstance(frame, str):
        frame = imread(frame)
        image_format='RGB'
    bbs, ccs, nimgs = do_detect_in_image(frame, det, image_format=image_format)
    # bbs, nimgs = do_detect_in_image(frame, det)
    if len(bbs) == 0:
        return [], [], [], []

    emb_unk = face_model((nimgs - 127.5) * 0.0078125).numpy()
    emb_unk = normalize(emb_unk)
    dists = np.dot(embeddings, emb_unk.T).T
    rec_idx = dists.argmax(-1)
    rec_dist = [dists[id, ii] for id, ii in enumerate(rec_idx)]
    rec_class = [image_class_name[ii] if dist > dist_thresh else "Unknown" for dist, ii in zip(rec_dist, rec_idx)]
    
    return rec_dist, rec_class, bbs, ccs

def draw_polyboxes(frame, rec_dist, rec_class, bbs, dist_thresh=0.6):
    for dist, label, bb in zip(rec_dist, rec_class, bbs):
        color = (0, 255, 0) if label != "Unknown" else (0, 0, 255)
        left, up, right, down = bb
        cv2.rectangle(frame, (left, up), (right, down), color, 2)
        cv2.putText(frame, f"Label: {label}, dist: {dist:.4f}", (left, up - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return frame


det, face_model, image_classes, image_class_name, embeddings = init_det_and_emb_model(model_file, known_user)

@app.route('/recognize', methods=['POST'])
@cross_origin(origin='*')
def recognize():
    # Check if the request contains a file
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    
    # Read the image file
    try:
        image = imread(file)
    except Exception as e:
        return jsonify({"error": f"Failed to read image: {e}"}), 400
    
    # Set the distance threshold
    dist_thresh = request.form.get('dist_thresh', default=0.6, type=float)
    
    # Recognize faces in the image
    rec_dist, rec_class, bbs, ccs = recognize_image(det, face_model, image_class_name, embeddings, image, dist_thresh)
    
    # Prepare the response
    response = {
        "recognized_faces": [
            {
                "label": label,
                "distance": round(float(dist), 2),
            }
            for label, dist, bb in zip(rec_class, rec_dist, bbs)
        ]
    }

    print(response)
    
    return jsonify(response)

# Run the Flask app
if __name__ == "__main__":    
    print("run app...")
    app.run(host="0.0.0.0", port=5000, debug=True)


# model_file = "models/GhostFaceNet_W1.3_S2_ArcFace.h5"
# embeddings_file = "known_user/image-dataset_aligned_112_112_embedding.npz"
# image_path = 'check/hazman.jpg'
# dist_thresh = 0.6
# known_user = "known_user/embeddings.npz"
# det, face_model = init_det_and_emb_model(model_file)
# image_classes, image_class_name, embeddings, embbeddings_file_path = read_embedding_file(known_user)
# frame = imread(image_path)
# rec_dist, rec_class, bbs, ccs = recognize_image(det, face_model, image_classes, image_class_name, embeddings, frame, dist_thresh)
# print(rec_class)
# print(rec_dist)
# embeddings_file = "known_user/embeddings.npz"
# known_user = '/kaggle/working/image-dataset_aligned_112_112/'
# image_classes, image_class_name, embeddings = embedding_images(det, face_model, known_user)
# rec_dist, rec_class, bbs = recognize_image(det, face_model, image_classes, embeddings, frame, dist_thresh)