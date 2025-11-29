import pygame
import numpy as np
try:
    from utils import EMASmoother
except ImportError:
    from src.utils import EMASmoother

class Avatar:
    def __init__(self, screen_width, screen_height):
        self.width = screen_width
        self.height = screen_height
        self.smoother = EMASmoother(alpha=0.6) # Balanced smoothing
        
        # Style configuration
        self.body_color = (255, 255, 255) # White core
        self.glow_color = (0, 255, 255)   # Cyan glow
        self.joint_color = (0, 200, 255)
        self.shoe_color = (255, 50, 50)   # Red shoes for style
        
        self.stroke_width = 10  # Nice thick stickman
        
    def _to_screen(self, norm_pt):
        """Converts normalized (x,y,z) to screen (x,y)."""
        if norm_pt is None:
            return None
        return (int(norm_pt[0] * self.width), int(norm_pt[1] * self.height))

    def draw_rounded_line(self, surface, start, end, color, width):
        """Draws a line with rounded caps."""
        pygame.draw.line(surface, color, start, end, width)
        pygame.draw.circle(surface, color, start, width // 2)
        pygame.draw.circle(surface, color, end, width // 2)

    def draw_neon_stick_limb(self, surface, start, end, width):
        """Draws a stickman limb with a glow."""
        # Outer Glow
        self.draw_rounded_line(surface, start, end, (0, 100, 100), width + 6)
        # Core
        self.draw_rounded_line(surface, start, end, self.body_color, width)

    def update_and_draw(self, surface, data, volume):
        if not data or not data.get("pose"):
            return

        pose = data["pose"]
        face = data.get("face")
        
        smoothed_pose = {}
        for key, val in pose.items():
            s_val = self.smoother.update(f"pose_{key}", (val[0], val[1]))
            smoothed_pose[key] = s_val

        # --- DRAW BODY (Stickman Style) ---
        structure = [
            ("left_shoulder", "right_shoulder"),
            ("left_shoulder", "left_hip"),
            ("right_shoulder", "right_hip"),
            ("left_shoulder", "left_elbow"),
            ("left_elbow", "left_wrist"),
            ("right_shoulder", "right_elbow"),
            ("right_elbow", "right_wrist"),
            ("left_hip", "right_hip"),
            ("left_hip", "left_knee"),
            ("left_knee", "left_ankle"),
            ("right_hip", "right_knee"),
            ("right_knee", "right_ankle"),
            # Neck
            ("neck", "left_shoulder"),
            ("neck", "right_shoulder"),
        ]
        
        # If neck is missing in smoothed (rare), calculate it
        if "neck" not in smoothed_pose and "left_shoulder" in smoothed_pose and "right_shoulder" in smoothed_pose:
             smoothed_pose["neck"] = (np.array(smoothed_pose["left_shoulder"]) + np.array(smoothed_pose["right_shoulder"])) / 2

        # Draw Limbs
        for p1_name, p2_name in structure:
            if p1_name in smoothed_pose and p2_name in smoothed_pose:
                start = self._to_screen(smoothed_pose[p1_name])
                end = self._to_screen(smoothed_pose[p2_name])
                self.draw_neon_stick_limb(surface, start, end, self.stroke_width)

        # --- DRAW SHOES ---
        # We use ankle and foot_index to define the shoe
        for side in ["left", "right"]:
            ankle = smoothed_pose.get(f"{side}_ankle")
            foot = smoothed_pose.get(f"{side}_foot_index")
            
            if ankle is not None:
                ankle_pt = self._to_screen(ankle)
                if foot is not None:
                    foot_pt = self._to_screen(foot)
                    # Draw oval along the vector from ankle to foot
                    # Simplified: Draw a large oval at the foot index
                    pygame.draw.circle(surface, (50, 0, 0), foot_pt, 18) # Darker sole
                    pygame.draw.circle(surface, self.shoe_color, foot_pt, 14) # Shoe
                    # Connect ankle to shoe
                    pygame.draw.line(surface, self.body_color, ankle_pt, foot_pt, 8)
                else:
                    # Fallback: Just a circle at ankle
                    pygame.draw.circle(surface, self.shoe_color, ankle_pt, 15)

        # --- DRAW HEAD (Dynamic Size) ---
        # Calculate head bounds based on Face Landmarks to prevent clipping
        head_center = None
        head_radius = 40 # Default
        
        # Gather all face points to find the bounding box
        all_face_pts = []
        if face:
            for key in ["left_eye", "right_eye", "lips", "jaw"]:
                pts = face.get(key)
                if pts: all_face_pts.extend(pts)
        
        if all_face_pts:
            # Project all to screen
            screen_face_pts = np.array([self._to_screen(pt) for pt in all_face_pts])
            min_x, min_y = np.min(screen_face_pts, axis=0)
            max_x, max_y = np.max(screen_face_pts, axis=0)
            
            # Center is mid of bounds
            center_x = int((min_x + max_x) / 2)
            center_y = int((min_y + max_y) / 2)
            head_center = (center_x, center_y)
            
            # Radius is half max dimension + padding
            max_dim = max(max_x - min_x, max_y - min_y)
            head_radius = int(max_dim / 2 * 1.4) # 1.4x padding
            
            # Clamp radius to sane limits
            head_radius = max(35, min(head_radius, 120))
        else:
            # Fallback if face tracking lost but pose exists
            if "nose" in smoothed_pose:
                head_center = self._to_screen(smoothed_pose["nose"])
            elif "neck" in smoothed_pose:
                 n = self._to_screen(smoothed_pose["neck"])
                 head_center = (n[0], n[1] - 50)

        if head_center:
            # Draw Neck connection first
            if "neck" in smoothed_pose:
                 pygame.draw.line(surface, self.body_color, self._to_screen(smoothed_pose["neck"]), head_center, 10)

            # Draw Head Circle
            pygame.draw.circle(surface, (0, 50, 50), head_center, head_radius + 4) # Glow
            pygame.draw.circle(surface, (20, 20, 25), head_center, head_radius) # Dark Face Background
            pygame.draw.circle(surface, self.body_color, head_center, head_radius, 3) # Outline

            # Draw Face Features
            if face:
                # Eyes & Blink
                blink_thresh = 0.18
                
                for side in ["left", "right"]:
                    blink = face.get(f"{side}_blink", 0.3)
                    iris = face.get(f"{side}_iris")
                    eye_pts = face.get(f"{side}_eye")
                    
                    if eye_pts:
                        # Calculate eye center relative to head logic? 
                        # No, use absolute tracking but ensure it's drawn on top
                        # We just draw them. Since head_circle encompasses them, they should be inside.
                        
                        eye_center_raw = np.mean(np.array(eye_pts)[:,:2], axis=0)
                        s_eye_center = self.smoother.update(f"{side}_eye_center", eye_center_raw)
                        eye_pos = self._to_screen(s_eye_center)
                        
                        if blink > blink_thresh:
                             # Open
                             pygame.draw.circle(surface, (255, 255, 255), eye_pos, 10) # Sclera
                             if iris:
                                 i_raw = np.mean(np.array(iris)[:,:2], axis=0)
                                 s_i = self.smoother.update(f"{side}_iris_pt", i_raw)
                                 pygame.draw.circle(surface, (0, 0, 0), self._to_screen(s_i), 4) # Pupil
                             else:
                                 pygame.draw.circle(surface, (0, 0, 0), eye_pos, 4)
                        else:
                             # Closed
                             pygame.draw.line(surface, self.body_color, (eye_pos[0]-8, eye_pos[1]), (eye_pos[0]+8, eye_pos[1]), 3)

                # Mouth
                lips = face.get("lips")
                if lips:
                    s_lips = []
                    for i, pt in enumerate(lips):
                        s_pt = self.smoother.update(f"lip_{i}", pt[:2])
                        s_lips.append(self._to_screen(s_pt))
                    
                    if len(s_lips) > 2:
                        pygame.draw.lines(surface, self.body_color, True, s_lips, 2)

        # --- HANDS (Thicker Fingers) ---
        hand_connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (5, 9), (9, 10), (10, 11), (11, 12),
            (9, 13), (13, 14), (14, 15), (15, 16),
            (13, 17), (17, 18), (18, 19), (19, 20),
            (0, 17)
        ]
        for hand_key in ["left_hand", "right_hand"]:
            hand_pts = data.get(hand_key)
            if hand_pts:
                s_hand = []
                for i, pt in enumerate(hand_pts):
                    s_pt = self.smoother.update(f"{hand_key}_{i}", (pt[0], pt[1]))
                    s_hand.append(self._to_screen(s_pt))
                
                # Draw connections
                for i, j in hand_connections:
                    if i < len(s_hand) and j < len(s_hand):
                        self.draw_rounded_line(surface, s_hand[i], s_hand[j], self.body_color, 4) # Thicker fingers
                
                # Draw joints/tips
                for pt in s_hand:
                     pygame.draw.circle(surface, self.glow_color, pt, 3)
