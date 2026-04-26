"""
Virtual Wig Try-On Application
A computer vision app for testing hairstyles using your webcam.

Uses classical Computer Vision techniques:
- Haar Cascade for face detection
- Haar Cascades for eyes; geometric region for nose
- Geometric calculations for wig positioning
- Multiple faces per frame supported

Controls:
    - Press 'N' or RIGHT ARROW: Next wig
    - Press 'P' or LEFT ARROW: Previous wig
    - Press '+'/'-': Increase / decrease wig scale
    - Press 'B': Toggle face bounding box
    - Press 'F': Toggle eyes / nose bounding boxes
    - Press 'M': Toggle mirror mode
    - Press 'Q' or ESC: Quit
"""

import cv2
import numpy as np
from src.wig_overlay import WigOverlay


class VirtualWigApp:
    """Main application for virtual wig try-on using webcam."""

    def __init__(self, camera_index: int = 0):
        """
        Initialize the Virtual Wig Try-On application.

        Args:
            camera_index: Index of the camera to use (default: 0).
        """
        self.camera_index = camera_index
        self.mirror_mode = True  # Mirror by default (like looking in a mirror)
        self.show_face_bbox = True
        self.show_feature_bbox = True  # Eyes, nose region

        cascade_dir = cv2.data.haarcascades
        # Face detection
        self.face_cascade = cv2.CascadeClassifier(
            cascade_dir + 'haarcascade_frontalface_default.xml'
        )
        # Eye cascade (head angle and feature drawing)
        self.eye_cascade = cv2.CascadeClassifier(
            cascade_dir + 'haarcascade_eye.xml'
        )

        # Initialize wig overlay
        self.wig_overlay = WigOverlay("assets/wigs")

        # User-controlled wig scale (applied on top of face-based scale)
        self.wig_scale_factor = 1.0
        self.wig_scale_min, self.wig_scale_max = 0.5, 2.0
        self.wig_scale_step = 0.1

        # Window name
        self.window_name = "Virtual Wig Try-On"

        # Smoothing for stable wig placement (one per face, by left-to-right index)
        self.prev_faces: list = []
        self.smooth_factor = 0.3

    def _draw_face_bbox(self, frame: np.ndarray, face_rect, rotation_angle: float | None = None) -> np.ndarray:
        """Draw a bounding box and basic debug info for the detected face."""
        if face_rect is None:
            return frame

        x, y, w, h = (int(v) for v in face_rect)
        color = (0, 255, 0)
        thickness = 2
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)

        label = "Face"
        if rotation_angle is not None:
            label = f"Face ({rotation_angle:+.1f}°)"

        text_org = (x, max(20, y - 8))
        cv2.putText(frame, label, text_org, cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        return frame

    def _detect_eyes(self, gray: np.ndarray, face_rect) -> list:
        """Detect eyes in face ROI. Returns list of (x, y, w, h) in frame coordinates."""
        x, y, w, h = (int(v) for v in face_rect)
        roi = gray[y : y + h // 2, x : x + w]
        if roi.size == 0:
            return []
        eyes = self.eye_cascade.detectMultiScale(
            roi, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20)
        )
        return [(x + ex, y + ey, ew, eh) for (ex, ey, ew, eh) in eyes]

    def _get_nose_region(self, face_rect) -> tuple:
        """Nose has no default Haar cascade; use geometric center strip of face. Returns (x, y, w, h)."""
        x, y, w, h = (int(v) for v in face_rect)
        # Center 40% width, vertical 35%–70% of face
        cw = int(w * 0.4)
        cx = x + (w - cw) // 2
        ny = y + int(h * 0.35)
        nh = int(h * 0.35)
        return (cx, ny, cw, nh)

    def _draw_feature_boxes(self, frame: np.ndarray, face_rect, gray: np.ndarray) -> np.ndarray:
        """Draw bounding boxes for eyes and nose region (CV only)."""
        if face_rect is None:
            return frame

        # Eyes (cyan)
        for (ex, ey, ew, eh) in self._detect_eyes(gray, face_rect):
            cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), (255, 255, 0), 2)
            cv2.putText(frame, "Eye", (ex, max(20, ey - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

        # Nose (geometric region, magenta)
        nx, ny, nw, nh = self._get_nose_region(face_rect)
        cv2.rectangle(frame, (nx, ny), (nx + nw, ny + nh), (255, 0, 255), 2)
        cv2.putText(frame, "Nose", (nx, max(20, ny - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)

        return frame

    def _smooth_position(self, current, previous):
        """Apply smoothing to reduce jitter."""
        if previous is None:
            return current
        return tuple(
            int(self.smooth_factor * c + (1 - self.smooth_factor) * p)
            for c, p in zip(current, previous)
        )

    def _detect_faces(self, gray_frame) -> list:
        """
        Detect all faces using Haar Cascade.

        Returns:
            List of (x, y, w, h), sorted left-to-right by center x.
        """
        faces = self.face_cascade.detectMultiScale(
            gray_frame,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(100, 100),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        # Sort left-to-right for stable smoothing across frames
        faces = sorted(faces, key=lambda f: f[0] + f[2] // 2)
        return list(faces)

    def _estimate_head_angle(self, gray_frame, face_rect):
        """
        Estimate head rotation angle using eye detection.

        Args:
            gray_frame: Grayscale frame.
            face_rect: (x, y, w, h) of detected face.

        Returns:
            Rotation angle in degrees.
        """
        x, y, w, h = face_rect

        # Focus on upper half of face for eye detection
        roi_gray = gray_frame[y:y + h // 2, x:x + w]

        eyes = self.eye_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(20, 20)
        )

        if len(eyes) >= 2:
            # Sort eyes by x position (left to right)
            eyes = sorted(eyes, key=lambda e: e[0])

            # Get center of first two eyes
            eye1_center = (eyes[0][0] + eyes[0][2] // 2,
                           eyes[0][1] + eyes[0][3] // 2)
            eye2_center = (eyes[1][0] + eyes[1][2] // 2,
                           eyes[1][1] + eyes[1][3] // 2)

            # Calculate angle between eyes
            delta_x = eye2_center[0] - eye1_center[0]
            delta_y = eye2_center[1] - eye1_center[1]

            if delta_x != 0:
                angle = np.degrees(np.arctan2(delta_y, delta_x))
                return angle

        return 0  # No rotation if eyes not detected

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame: detect all faces, overlay wig on each, draw bboxes.
        """
        if self.mirror_mode:
            frame = cv2.flip(frame, 1)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._detect_faces(gray)

        # Smooth each face by left-to-right index
        smoothed = []
        for i, (x, y, w, h) in enumerate(faces):
            current = (x, y, w, h)
            if i < len(self.prev_faces):
                x, y, w, h = self._smooth_position(current, self.prev_faces[i])
            smoothed.append((x, y, w, h))
        self.prev_faces = smoothed

        for (x, y, w, h) in smoothed:
            forehead_x = x + w // 2
            forehead_y = y
            rotation = self._estimate_head_angle(gray, (x, y, w, h))

            wig_data = self.wig_overlay.get_current_wig()
            category = wig_data.get("category", "men") if wig_data else "men"
            frame = self.wig_overlay.overlay_wig(
                frame=frame,
                forehead_center=(forehead_x, forehead_y),
                face_width=w,
                face_top=y,
                rotation_angle=rotation,
                scale_factor=self.wig_scale_factor,
                hair_category=category,
            )

            if self.show_face_bbox:
                frame = self._draw_face_bbox(frame, (x, y, w, h), rotation_angle=rotation)
            if self.show_feature_bbox:
                frame = self._draw_feature_boxes(frame, (x, y, w, h), gray)

        return frame

    def _draw_ui(self, frame: np.ndarray) -> np.ndarray:
        """Draw UI elements on the frame."""
        # Semi-transparent overlay for text background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (480, 120), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)

        # Current wig name, category, scale
        wig = self.wig_overlay.get_current_wig()
        wig_name = wig["name"] if wig else "No wigs loaded"
        wig_cat = wig.get("category", "—") if wig else "—"
        wig_count = len(self.wig_overlay.wigs)
        current_idx = self.wig_overlay.current_wig_index + 1 if wig_count > 0 else 0

        cv2.putText(frame, f"Wig: {wig_name} ({current_idx}/{wig_count}) [{wig_cat}]",
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Scale: {self.wig_scale_factor:.1f} [+/-]",
                    (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)

        # Controls hint
        mirror_status = "ON" if self.mirror_mode else "OFF"
        bbox_status = "ON" if self.show_face_bbox else "OFF"
        feat_status = "ON" if self.show_feature_bbox else "OFF"
        cv2.putText(frame, f"[N/P] Wig | [B] Face: {bbox_status} | [F] Eyes/Nose: {feat_status}",
                    (20, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(frame, f"[M] Mirror: {mirror_status} | [Q/ESC] Quit",
                    (20, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        return frame

    def run(self):
        """Run the main application loop."""
        print("\n=== Virtual Wig Try-On ===")
        print("Using Classical Computer Vision (Haar Cascades)")
        print("Starting webcam...")

        # Open webcam
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            print("Please check if your camera is connected and not in use.")
            return

        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)

        print("Webcam started successfully!")
        print("\nControls:")
        print("  N / RIGHT ARROW : Next wig")
        print("  P / LEFT ARROW  : Previous wig")
        print("  + / -           : Wig scale up / down")
        print("  B               : Toggle face bounding box")
        print("  F               : Toggle eyes / nose boxes")
        print("  M               : Toggle mirror mode")
        print("  Q / ESC         : Quit")
        print("\nLook at the camera and try different wigs!\n")

        # Create window
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Failed to capture frame.")
                    break

                # Process frame (face detection + wig overlay)
                frame = self._process_frame(frame)

                # Draw UI
                frame = self._draw_ui(frame)

                # Display frame
                cv2.imshow(self.window_name, frame)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q') or key == 27:  # Q or ESC
                    print("Exiting...")
                    break
                elif key == ord('n') or key == 83:  # N or RIGHT arrow
                    self.wig_overlay.next_wig()
                    wig = self.wig_overlay.get_current_wig()
                    if wig:
                        print(f"Switched to: {wig['name']}")
                elif key == ord('p') or key == 81:  # P or LEFT arrow
                    self.wig_overlay.previous_wig()
                    wig = self.wig_overlay.get_current_wig()
                    if wig:
                        print(f"Switched to: {wig['name']}")
                elif key == ord('m'):  # M - toggle mirror
                    self.mirror_mode = not self.mirror_mode
                    print(f"Mirror mode: {'ON' if self.mirror_mode else 'OFF'}")
                elif key == ord('b'):  # B - toggle face bounding box
                    self.show_face_bbox = not self.show_face_bbox
                    print(f"Face bounding box: {'ON' if self.show_face_bbox else 'OFF'}")
                elif key == ord('f'):  # F - toggle eyes/nose boxes
                    self.show_feature_bbox = not self.show_feature_bbox
                    print(f"Eyes/Nose boxes: {'ON' if self.show_feature_bbox else 'OFF'}")
                elif key in (ord('+'), ord('=')):  # + or = scale up
                    self.wig_scale_factor = min(self.wig_scale_max, self.wig_scale_factor + self.wig_scale_step)
                    print(f"Wig scale: {self.wig_scale_factor:.1f}")
                elif key == ord('-'):  # - scale down
                    self.wig_scale_factor = max(self.wig_scale_min, self.wig_scale_factor - self.wig_scale_step)
                    print(f"Wig scale: {self.wig_scale_factor:.1f}")

        finally:
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()
            print("Application closed.")


def main():
    """Entry point for the application."""
    app = VirtualWigApp(camera_index=0)
    app.run()


if __name__ == "__main__":
    main()
