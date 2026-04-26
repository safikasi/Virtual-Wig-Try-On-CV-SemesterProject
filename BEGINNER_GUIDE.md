# Virtual Wig Try-On — Explained for Beginners

This guide explains what this project does and how it works, using simple words and step-by-step ideas.

---

## What Does This Project Do?

Imagine standing in front of your webcam and seeing yourself on the screen **with different hairstyles on your head** — without actually wearing a wig. You can switch between styles with the keyboard. That’s exactly what this app does.

**In one sentence:**  
The app uses your webcam, finds your face, and draws a “wig” image on top of your head so you can try different looks in real time.

---

## Big Picture: What Happens Step by Step

1. **Turn on the webcam**  
   The program opens your camera and gets a continuous video (many images per second).

2. **Find your face**  
   In each image (frame), it looks for a face. It uses a built-in “face finder” (Haar Cascade) that was trained to recognize faces. No AI training needed on your side.

3. **Figure out where to put the wig**  
   Once it knows where the face is (position and size), it decides:
   - How big the wig image should be (based on face size)
   - Where to place it (roughly above the forehead, centered on the face)

4. **Optional: Detect head tilt**  
   It can also find your eyes and use them to see if your head is tilted. If it is, it rotates the wig image so it looks like it’s really on your head.

5. **Draw the wig on the image**  
   It takes a wig image (a PNG with a transparent background), resizes and rotates it, then “pastes” it onto the video frame so it looks like you’re wearing it.

6. **Show the result**  
   The final image (with the wig drawn on) is shown in a window. This repeats for every frame, so you see a live “virtual try-on.”

---

## Main Ideas (Simple Explanations)

### 1. What is “face detection”?

**Face detection** means: *“Find where a face is in a picture.”*

The program gets a rectangle around the face: position (x, y) and size (width, height). It uses OpenCV’s **Haar Cascade** — a classic method that looks for patterns that look like faces (e.g. eyes, nose, face outline). No neural network is used here.

### 2. What is “wig overlay”?

**Overlay** means: *“Put one image on top of another.”*

- **Bottom image:** the current frame from the webcam (you in the room).
- **Top image:** the wig (a PNG with transparent parts).

Where it’s not transparent, the wig “covers” the video; where it’s transparent, you still see the video. So it looks like the wig is on your head.

### 3. What is “alpha” or “transparency”?

Many images are fully opaque (you can’t see through them). PNG wig images can have an **alpha channel**: each pixel can be “a bit see-through.”  
The program uses this to blend the wig with the video so edges don’t look like a hard cutout.

### 4. Why “mirror mode”?

By default the image is flipped horizontally (mirror mode). So when you move left, your image moves left on screen — like looking in a mirror. That feels more natural for trying on a wig.

---

## How the Code Is Organized

| Part | File / folder | What it does (simple) |
|------|----------------|-----------------------|
| **Start the app** | `main.py` | Opens the webcam, runs the loop, handles keys (next/previous wig, mirror, quit). |
| **Face & head** | `main.py` (inside `VirtualWigApp`) | Finds the face, smooths its position, finds eyes and estimates head tilt. |
| **Wig logic** | `src/wig_overlay.py` | Loads wig images from a folder, resizes/rotates the current wig, and draws it on the frame. |
| **Wig images** | `assets/wigs/` | PNG files (e.g. `short_black.png`, `long_blonde.png`). The app lists all PNGs here and lets you cycle through them. |
| **Optional** | `generate_sample_wigs.py` | Creates example wig images if you don’t have any. |

---

## Flow (In Simple Terms)

```
Webcam → Get one frame (image)
    ↓
Convert to grayscale (for face detection)
    ↓
Run face detector → Get face position and size
    ↓
(Optional) Run eye detector → Get head rotation angle
    ↓
Load current wig image from assets/wigs/
    ↓
Resize wig to match face size, rotate if head is tilted
    ↓
Place wig above forehead, centered on face
    ↓
Blend wig onto the frame (using transparency)
    ↓
Show the result in a window
    ↓
Repeat for the next frame (so it’s real-time)
```

---

## What You Need to Run It

- **Python** (e.g. 3.8 or newer)
- **Libraries:** OpenCV (`opencv-python`) and NumPy (see `requirements.txt`)
- **Webcam**
- **Wig images:** Put PNG files with transparent backgrounds in `assets/wigs/`, or run `generate_sample_wigs.py` to create samples

---

## Controls (Quick Reference)

| Key | Action |
|-----|--------|
| `N` or `→` | Next wig |
| `P` or `←` | Previous wig |
| `M` | Toggle mirror on/off |
| `Q` or `ESC` | Quit |

---

## Summary

- The project is a **real-time virtual wig try-on**: webcam + face detection + wig overlay.
- It uses **classical computer vision** (Haar Cascades for face and eyes), not deep learning.
- **`main.py`** runs the app and does face/eye detection; **`src/wig_overlay.py`** loads wigs and draws them on each frame; **`assets/wigs/`** holds the wig PNGs.
- You see yourself with different hairstyles and can switch them with the keyboard.

If you’re new to code, start by reading the comments in `main.py` and `src/wig_overlay.py`, then run `python main.py` and try the keys above.
