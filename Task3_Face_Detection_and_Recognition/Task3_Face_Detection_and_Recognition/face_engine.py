import os
import cv2
import numpy as np


class FaceEngine:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))

        self.detector_model = os.path.join(
            self.base_dir,
            "models",
            "face_detection_yunet_2023mar.onnx"
        )

        self.recognizer_model = os.path.join(
            self.base_dir,
            "models",
            "face_recognition_sface_2021dec.onnx"
        )

        self.known_faces_dir = os.path.join(
            self.base_dir,
            "known_faces"
        )

        self.cosine_threshold = 0.363

        self.detector = cv2.FaceDetectorYN.create(
            self.detector_model,
            "",
            (320, 320),
            0.5,
            0.3,
            5000
        )

        self.recognizer = cv2.FaceRecognizerSF.create(
            self.recognizer_model,
            ""
        )

        self.known_features = {}

        self.last_status = {
            "face_count": 0,
            "recognized": 0,
            "unknown": 0,
            "known_faces": 0
        }

        self.load_known_faces()

    def detect_faces(self, frame):
        height, width = frame.shape[:2]

        self.detector.setInputSize((width, height))

        result = self.detector.detect(frame)

        if result is None:
            return []

        _, faces = result

        if faces is None:
            return []

        return faces

    def extract_feature(self, frame, face):
        aligned_face = self.recognizer.alignCrop(frame, face)
        feature = self.recognizer.feature(aligned_face)

        return feature

    def load_known_faces(self):
        self.known_features = {}

        if not os.path.exists(self.known_faces_dir):
            os.makedirs(self.known_faces_dir)

        for filename in os.listdir(self.known_faces_dir):

            if not filename.lower().endswith(
                (".jpg", ".jpeg", ".png")
            ):
                continue

            image_path = os.path.join(
                self.known_faces_dir,
                filename
            )

            image = cv2.imread(image_path)

            if image is None:
                continue

            faces = self.detect_faces(image)

            if faces is None or len(faces) == 0:
                continue

            largest_face = max(
                faces,
                key=lambda face: face[2] * face[3]
            )

            try:
                feature = self.extract_feature(
                    image,
                    largest_face
                )

                person_name = os.path.splitext(filename)[0]

                self.known_features[person_name] = feature

            except Exception:
                continue

        self.last_status["known_faces"] = len(
            self.known_features
        )

    def recognize_face(self, frame, face):
        if not self.known_features:
            return "Unknown", 0.0

        try:
            feature = self.extract_feature(
                frame,
                face
            )

            best_name = "Unknown"
            best_score = -1.0

            for name, known_feature in self.known_features.items():

                score = self.recognizer.match(
                    feature,
                    known_feature,
                    cv2.FaceRecognizerSF_FR_COSINE
                )

                if score > best_score:
                    best_score = score
                    best_name = name

            if best_score >= self.cosine_threshold:
                return best_name, float(best_score)

            return "Unknown", float(best_score)

        except Exception:
            return "Unknown", 0.0

    def process_frame(self, frame):
        faces = self.detect_faces(frame)

        recognized_count = 0
        unknown_count = 0

        for face in faces:

            x, y, w, h = face[:4].astype(int)

            name, score = self.recognize_face(
                frame,
                face
            )

            if name == "Unknown":
                unknown_count += 1
                label = "Unknown"
            else:
                recognized_count += 1
                label = f"{name}  {score:.2f}"

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (255, 190, 0),
                2
            )

            cv2.rectangle(
                frame,
                (x, max(0, y - 32)),
                (x + w, y),
                (255, 190, 0),
                -1
            )

            cv2.putText(
                frame,
                label,
                (x + 8, y - 9),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                2,
                cv2.LINE_AA
            )

        self.last_status = {
            "face_count": len(faces),
            "recognized": recognized_count,
            "unknown": unknown_count,
            "known_faces": len(self.known_features)
        }

        cv2.rectangle(
            frame,
            (0, 0),
            (frame.shape[1], 48),
            (20, 20, 20),
            -1
        )

        status_text = (
            f"Faces: {len(faces)}   "
            f"Recognized: {recognized_count}   "
            f"Unknown: {unknown_count}"
        )

        cv2.putText(
            frame,
            status_text,
            (18, 31),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        return frame

    def get_status(self):
        return self.last_status.copy()