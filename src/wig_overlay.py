"""
Wig Overlay Module
Handles loading and overlaying wig images onto detected faces.
Supports hair categories (men, woman, long) and head rotation with correct anchoring.
"""

import cv2
import numpy as np
from pathlib import Path

# Category: anchor_y_fraction (where wig meets forehead, 0=top 1=bottom), width_mult, y_offset_above_forehead
CATEGORY_PARAMS = {
    "men": {"anchor_y": 0.92, "width_mult": 1.85, "y_offset_ratio": 0.42},
    "woman": {"anchor_y": 0.88, "width_mult": 1.9, "y_offset_ratio": 0.45},
    "long": {"anchor_y": 0.32, "width_mult": 2.0, "y_offset_ratio": 0.15},
}


def _infer_category(name: str) -> str:
    """Infer hair category from wig filename (e.g. long_black -> long, short_* -> men)."""
    n = name.lower()
    if n.startswith("long_"):
        return "long"
    if n.startswith("curly_"):
        return "woman"
    if any(n.startswith(p) for p in ("short_", "spiky_", "afro_")):
        return "men"
    return "men"


class WigOverlay:
    """Handles wig image loading and overlay operations."""

    def __init__(self, wigs_folder: str = "assets/wigs"):
        """
        Initialize the WigOverlay with a folder of wig images.

        Args:
            wigs_folder: Path to folder containing wig PNG images with transparency.
        """
        self.wigs_folder = Path(wigs_folder)
        self.wigs = []
        self.current_wig_index = 0
        self._load_wigs()

    def _load_wigs(self):
        """Load all PNG wig images and assign category from filename."""
        if not self.wigs_folder.exists():
            print(f"Warning: Wigs folder '{self.wigs_folder}' not found.")
            return

        for wig_path in sorted(self.wigs_folder.glob("*.png")):
            wig_img = cv2.imread(str(wig_path), cv2.IMREAD_UNCHANGED)
            if wig_img is not None:
                if wig_img.ndim == 2:
                    wig_img = cv2.cvtColor(wig_img, cv2.COLOR_GRAY2BGRA)
                elif wig_img.shape[2] == 3:
                    alpha = np.ones(wig_img.shape[:2], dtype=wig_img.dtype) * 255
                    wig_img = np.dstack([wig_img, alpha])
                name = wig_path.stem
                category = _infer_category(name)
                self.wigs.append({
                    "name": name,
                    "image": wig_img,
                    "category": category,
                })
                print(f"Loaded wig: {wig_path.name} [{category}]")

        if not self.wigs:
            print("No wig images found. Add PNG images to the assets/wigs folder.")

    def get_current_wig(self) -> dict:
        """Get the currently selected wig."""
        if not self.wigs:
            return None
        return self.wigs[self.current_wig_index]

    def next_wig(self):
        """Switch to the next wig in the collection."""
        if self.wigs:
            self.current_wig_index = (self.current_wig_index + 1) % len(self.wigs)

    def previous_wig(self):
        """Switch to the previous wig in the collection."""
        if self.wigs:
            self.current_wig_index = (self.current_wig_index - 1) % len(self.wigs)

    def overlay_wig(
        self,
        frame: np.ndarray,
        forehead_center: tuple,
        face_width: int,
        face_top: int,
        rotation_angle: float = 0,
        scale_factor: float = 1.0,
        hair_category: str | None = None,
    ) -> np.ndarray:
        """
        Overlay the current wig onto the frame. Rotation is around an anchor point
        (where the wig meets the forehead) so the wig stays attached when the head tilts.

        Args:
            frame: The video frame (BGR format).
            forehead_center: (x, y) position of the forehead center.
            face_width: Width of the detected face for scaling.
            face_top: Y coordinate of the top of the face.
            rotation_angle: Head tilt in degrees (positive = right ear down).
            scale_factor: User scale multiplier (e.g. 0.8–1.5).
            hair_category: "men", "woman", or "long"; uses current wig category if None.

        Returns:
            Frame with wig overlaid.
        """
        wig_data = self.get_current_wig()
        if wig_data is None:
            return frame

        wig_img = wig_data["image"].copy()
        category = hair_category or wig_data.get("category", "men")
        params = CATEGORY_PARAMS.get(category, CATEGORY_PARAMS["men"])
        anchor_y_frac = params["anchor_y"]
        width_mult = params["width_mult"]
        y_offset_ratio = params["y_offset_ratio"]

        # Base scale from face width and category, then user scale
        wig_scale = (face_width * width_mult / wig_img.shape[1]) * scale_factor
        new_width = max(1, int(wig_img.shape[1] * wig_scale))
        new_height = max(1, int(wig_img.shape[0] * wig_scale))
        wig_resized = cv2.resize(wig_img, (new_width, new_height), interpolation=cv2.INTER_AREA)

        # Anchor: point on the wig that should stay at the forehead (bottom-center for short, higher for long)
        anchor_x = new_width / 2.0
        anchor_y = new_height * anchor_y_frac

        if abs(rotation_angle) >= 0.5:
            # Rotate around the anchor so the wig tilts with the head
            M = cv2.getRotationMatrix2D((anchor_x, anchor_y), rotation_angle, 1.0)
            # Bbox of rotated rectangle (0,0)-(new_width, new_height)
            corners = np.array([
                [0, 0], [new_width, 0], [new_width, new_height], [0, new_height]
            ], dtype=np.float32)
            rotated = (M[:, :2] @ corners.T).T + np.array([M[0, 2], M[1, 2]])
            min_x, min_y = rotated.min(axis=0)
            max_x, max_y = rotated.max(axis=0)
            new_w = int(np.ceil(max_x - min_x))
            new_h = int(np.ceil(max_y - min_y))
            # Shift so bbox starts at (0,0)
            M[0, 2] -= min_x
            M[1, 2] -= min_y
            wig_resized = cv2.warpAffine(
                wig_resized, M, (new_w, new_h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0, 0),
            )
            # Where did the anchor land in the rotated image?
            anchor_dst = M[:, :2] @ np.array([anchor_x, anchor_y]) + np.array([M[0, 2], M[1, 2]])
            place_x = int(forehead_center[0] - anchor_dst[0])
            place_y = int(forehead_center[1] - anchor_dst[1])
        else:
            # No rotation: place so anchor is at forehead_center; small nudge so hairline sits above forehead
            place_x = int(forehead_center[0] - anchor_x)
            place_y = int(forehead_center[1] - anchor_y)
            place_y -= int(new_height * y_offset_ratio * 0.3)

        return self._blend_overlay(frame, wig_resized, place_x, place_y)

    def _blend_overlay(self, background: np.ndarray, overlay: np.ndarray,
                       x: int, y: int) -> np.ndarray:
        """Blend an RGBA overlay onto a BGR background at position (x, y)."""
        result = background.copy()
        h, w = overlay.shape[:2]
        bg_h, bg_w = background.shape[:2]

        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(bg_w, x + w), min(bg_h, y + h)
        ox1 = x1 - x
        oy1 = y1 - y
        ox2 = ox1 + (x2 - x1)
        oy2 = oy1 + (y2 - y1)

        if x2 <= x1 or y2 <= y1:
            return result

        overlay_region = overlay[oy1:oy2, ox1:ox2]
        bg_region = result[y1:y2, x1:x2]
        alpha = overlay_region[:, :, 3:4] / 255.0
        blended = (alpha * overlay_region[:, :, :3] + (1 - alpha) * bg_region).astype(np.uint8)
        result[y1:y2, x1:x2] = blended
        return result
