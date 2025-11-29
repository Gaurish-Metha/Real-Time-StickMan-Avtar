import mediapipe as mp
import cv2
import numpy as np

class HolisticTracker:
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            refine_face_landmarks=True,  # CRITICAL for iris/lips
            model_complexity=1  # Revert to 1 (Balanced) for better accuracy
        )

    def process(self, frame):
        """
        Process a BGR frame and return the raw MediaPipe results.
        """
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = self.holistic.process(image_rgb)
        image_rgb.flags.writeable = True
        return results

    def _get_blink_ratio(self, eye_points, landmarks):
        """Calculates Eye Aspect Ratio (EAR) to detect blinking."""
        # Vertical distances
        # indices: 0=left_corner, 1=top_right, 2=top_left, 3=right_corner, 4=bottom_left, 5=bottom_right
        # mapped from the specific indices below
        # We need to extract specific points.
        # eye_indices layout in our extract_landmarks: 
        # Left Eye: [33, 160, 158, 133, 153, 144] (approx)
        # Let's use the raw points passed in.
        
        # P2-P6 (vertical 1)
        p2 = np.array(eye_points[1])
        p6 = np.array(eye_points[5])
        # P3-P5 (vertical 2)
        p3 = np.array(eye_points[2])
        p5 = np.array(eye_points[4])
        # P1-P4 (horizontal)
        p1 = np.array(eye_points[0])
        p4 = np.array(eye_points[3])

        vertical_dist = np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)
        horizontal_dist = np.linalg.norm(p1 - p4)
        
        if horizontal_dist == 0: return 0.0
        return vertical_dist / (2.0 * horizontal_dist)

    def extract_landmarks(self, results):
        """
        Parses the MediaPipe results into a structured dictionary.
        Returns None if no pose is detected.
        """
        if not results.pose_landmarks:
            return None

        data = {
            "pose": {},
            "face": {},
            "left_hand": {},
            "right_hand": {}
        }

        # --- POSE ---
        pl = results.pose_landmarks.landmark
        mp_pose = mp.solutions.pose.PoseLandmark
        
        def get_lm(idx):
            lm = pl[idx]
            return (lm.x, lm.y, lm.z, lm.visibility)

        data["pose"] = {
            "nose": get_lm(mp_pose.NOSE),
            "left_shoulder": get_lm(mp_pose.LEFT_SHOULDER),
            "right_shoulder": get_lm(mp_pose.RIGHT_SHOULDER),
            "left_elbow": get_lm(mp_pose.LEFT_ELBOW),
            "right_elbow": get_lm(mp_pose.RIGHT_ELBOW),
            "left_wrist": get_lm(mp_pose.LEFT_WRIST),
            "right_wrist": get_lm(mp_pose.RIGHT_WRIST),
            "left_hip": get_lm(mp_pose.LEFT_HIP),
            "right_hip": get_lm(mp_pose.RIGHT_HIP),
            "left_knee": get_lm(mp_pose.LEFT_KNEE),
            "right_knee": get_lm(mp_pose.RIGHT_KNEE),
            "left_ankle": get_lm(mp_pose.LEFT_ANKLE),
            "right_ankle": get_lm(mp_pose.RIGHT_ANKLE),
            "left_foot_index": get_lm(mp_pose.LEFT_FOOT_INDEX),
            "right_foot_index": get_lm(mp_pose.RIGHT_FOOT_INDEX),
            # Add neck approximation
            "neck": tuple(np.mean([np.array(get_lm(mp_pose.LEFT_SHOULDER)), np.array(get_lm(mp_pose.RIGHT_SHOULDER))], axis=0))
        }

        # --- HANDS ---
        def extract_hand(hand_landmarks):
            if not hand_landmarks:
                return None
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append((lm.x, lm.y, lm.z))
            return landmarks

        data["left_hand"] = extract_hand(results.left_hand_landmarks)
        data["right_hand"] = extract_hand(results.right_hand_landmarks)

        # --- FACE ---
        if results.face_landmarks:
            fl = results.face_landmarks.landmark
            
            # Detailed Lip Indices (Outer and Inner for better shape)
            # Outer: 61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95
            lips_indices = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
            
            # Eye Indices (Ordered for Blink Ratio)
            # P1, P2, P3, P4, P5, P6
            left_eye_indices = [33, 160, 158, 133, 153, 144]
            right_eye_indices = [362, 385, 387, 263, 373, 380]
            
            left_iris_indices = [468, 469, 470, 471]
            right_iris_indices = [473, 474, 475, 476]
            
            def get_face_lms(indices):
                return [(fl[i].x, fl[i].y, fl[i].z) for i in indices]

            left_eye_pts = get_face_lms(left_eye_indices)
            right_eye_pts = get_face_lms(right_eye_indices)

            data["face"] = {
                "lips": get_face_lms(lips_indices),
                "left_eye": left_eye_pts,
                "right_eye": right_eye_pts,
                "left_iris": get_face_lms(left_iris_indices),
                "right_iris": get_face_lms(right_iris_indices),
                "jaw": [(fl[152].x, fl[152].y, fl[152].z)],
                "left_blink": self._get_blink_ratio(left_eye_pts, fl),
                "right_blink": self._get_blink_ratio(right_eye_pts, fl)
            }
        
        return data
