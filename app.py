import time
from flask import Flask, request, jsonify
import tensorflow as tf
from flask_cors import CORS, cross_origin
import os
import cv2
import numpy as np
from skimage.io import imread
from sklearn.preprocessing import normalize
from face_detector import YoloV5FaceDetector
from deepface import DeepFace
import time
import logging

print("START")

# Initialize Flask app
app = Flask(__name__)
app.logger.setLevel(logging.INFO)
CORS(app)

# Load models and embeddings before running the Flask app
model_file = os.path.join("models", "GhostFaceNet_W1.3_S2_ArcFace.h5")
embeddings_file = os.path.join("known_user", "embeddings_new.npz")
known_user = os.path.join("known_user", "embeddings_new.npz")

def init_det_and_emb_model(model_file, embeddings_file):
    det = YoloV5FaceDetector()
    face_model = tf.keras.models.load_model(model_file, compile=False)
    data = np.load(embeddings_file)
    image_classes, image_class_name, embeddings = data["imm_classes"], data["imm_class_names"], data["embs"]
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
    print('Detecting face from image')
    imm_BGR = image if image_format == "BGR" else image[:, :, ::-1]
    imm_RGB = image[:, :, ::-1] if image_format == "BGR" else image
    bboxes, pps, ccs = det.__call__(imm_BGR)
    nimgs = face_align_landmarks_sk(imm_RGB, pps)
    bbs, ccs = bboxes[:, :4].astype("int"), bboxes[:, -1]
    return bbs, ccs, nimgs

def recognize_image(det, face_model, image_class_name, embeddings, frame, dist_thresh=0.6, image_format='BGR'):
    print('Starting face recognition')
    if isinstance(frame, str):
        frame = imread(frame)
        image_format='RGB'
    
    try:
        bbs, ccs, nimgs = do_detect_in_image(frame, det, image_format=image_format)
        if len(bbs) == 0:
            return [], [], [], []

        emb_unk = face_model((nimgs - 127.5) * 0.0078125).numpy()
        emb_unk = normalize(emb_unk)
        dists = np.dot(embeddings, emb_unk.T).T
        rec_idx = dists.argmax(-1)
        rec_dist = [dists[id, ii] for id, ii in enumerate(rec_idx)]
        rec_class = [image_class_name[ii] if dist > dist_thresh else "Unknown" for dist, ii in zip(rec_dist, rec_idx)]
        
        return rec_dist, rec_class, bbs, ccs
    except Exception as e:
        print(f"Error in recognize_image: {str(e)}")
        return [], [], [], []

# Initialize models
det, face_model, image_classes, image_class_name, embeddings = init_det_and_emb_model(model_file, known_user)

@app.route('/validate', methods=['POST'])
@cross_origin(origin='*')
def validate():
    start_time = time.time()
    app.logger.info("Received request for validation")
    
    if 'file' not in request.files:
        return jsonify({
            "success": False,
            "message": "No file provided",
            "liveness": False,
            "recognized_faces": []
        }), 400
    
    file = request.files['file']
    
    try:
        image = imread(file)
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Failed to read image: {e}",
            "liveness": False,
            "recognized_faces": []
        }), 400
    
    # Set the distance threshold
    dist_thresh = request.form.get('dist_thresh', default=0.6, type=float)
    
    # Liveness Detection
    try:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # liveness detection
        face_objs = DeepFace.extract_faces(
            img_path=image,
            detector_backend='opencv',
            enforce_detection=False,
            anti_spoofing=True
        )
        
        if not face_objs or not all(face_obj["is_real"] for face_obj in face_objs):
            res = {
                "success": False,
                "message": "Liveness check failed - spoofing detected",
                "liveness": False,
                "recognized_faces": []
            }
            app.logger.info("Liveness check failed - spoofing detected")
            app.logger.info(res)
            return jsonify(res), 400
            
    except Exception as e:
        app.logger.error(f"Liveness detection failed: {str(e)}")
        return jsonify({
            "success": False,
            "message": f"Liveness detection error: {str(e)}",
            "liveness": False,
            "recognized_faces": []
        }), 500
    
    # Face Recognition
    try:
        rec_dist, rec_class, bbs, ccs = recognize_image(
            det, face_model, image_class_name, embeddings, image, dist_thresh, 'RGB'
        )
        
        if not rec_class:
            return jsonify({
                "success": False,
                "message": "No faces detected",
                "liveness": True,
                "recognized_faces": []
            }), 200
        
        recognized_faces = []
        for label, dist, bb in zip(rec_class, rec_dist, bbs):
            recognized_faces.append({
                "label": label,
                "distance": float(dist),
                "bounding_box": {
                    "left": int(bb[0]),
                    "top": int(bb[1]),
                    "right": int(bb[2]),
                    "bottom": int(bb[3])
                }
            })
        
        known_faces = [face for face in recognized_faces if face["label"] != "Unknown"]
        
        response = {
            "success": True,
            "message": "Validation successful",
            "liveness": True,
            "recognized_faces": recognized_faces,
            "processing_time": round(time.time() - start_time, 2)
        }
        
        if known_faces:
            response["best_match"] = {
                "nim": known_faces[0]["label"],
                "confidence": known_faces[0]["distance"]
            }
        app.logger.info(f"Validation successful: {response}")
        return jsonify(response), 200
        
    except Exception as e:
        print(f"[ERROR] Face recognition failed: {str(e)}")
        return jsonify({
            "success": False,
            "message": f"Face recognition error: {str(e)}",
            "liveness": True,
            "recognized_faces": []
        }), 500

@app.route('/recognize', methods=['POST'])
@cross_origin(origin='*')
def recognize():
    """Legacy endpoint - use /validate instead"""
    return validate()

if __name__ == "__main__":
    print("Starting Face Recognition API...")
    app.run(host="0.0.0.0", port=5000, debug=True)