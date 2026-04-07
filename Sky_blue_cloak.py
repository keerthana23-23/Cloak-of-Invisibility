import cv2
import numpy as np
import time  # For unique filename

def get_background(cap, frames=60):
    """Capture a stable background frame by averaging multiple frames."""
    accumulated_bg = None
    valid_frames = 0
    for _ in range(frames):
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        frame = frame.astype(np.float32)
        if accumulated_bg is None:
            accumulated_bg = frame
        else:
            accumulated_bg += frame
        valid_frames += 1
    if valid_frames > 0:
        accumulated_bg /= valid_frames
    return accumulated_bg.astype(np.uint8)

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open webcam.")
        return

    print("➡️ Make sure the scene is empty, then press 'b' to capture background.")
    print("➡️ Press 'q' to quit.")

    background = None

    # ================= Video Recording Setup =================
    record_video = True  # Toggle recording on/off
    if record_video:
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # ✅ Universal, works on Windows
        filename = f"cloak_demo_{int(time.time())}.avi"  # Safe for WhatsApp
        out = cv2.VideoWriter(filename, fourcc, 20.0, (640,480))
        print(f"🎥 Recording will be saved as {filename}")
    # =========================================================

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)

        if background is None:
            cv2.putText(frame, "Press 'b' to capture background", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.imshow("Invisibility Cloak", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('b'):
                print("📸 Capturing background...")
                background = get_background(cap)
                print("✅ Background captured.")
            elif key == ord('q'):
                break
            continue

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # ================= Color Ranges =================
        # Red
        lower_red1 = np.array([0, 120, 70])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 120, 70])
        upper_red2 = np.array([180, 255, 255])
        mask_red = cv2.bitwise_or(cv2.inRange(hsv, lower_red1, upper_red1),
                                  cv2.inRange(hsv, lower_red2, upper_red2))

        # Sky Blue
        lower_blue = np.array([90, 50, 50])
        upper_blue = np.array([130, 255, 255])
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

        # Gulabi (Pink)
        lower_pink = np.array([140, 50, 50])
        upper_pink = np.array([170, 255, 255])
        mask_pink = cv2.inRange(hsv, lower_pink, upper_pink)

        mask = cv2.bitwise_or(mask_red, mask_blue)
        mask = cv2.bitwise_or(mask, mask_pink)
        # =================================================

        # Mask Cleaning
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                np.ones((3,3), np.uint8), iterations=2)
        mask = cv2.dilate(mask, np.ones((3,3), np.uint8), iterations=1)
        mask = cv2.GaussianBlur(mask, (5,5), 0)

        inverse_mask = cv2.bitwise_not(mask)

        current_no_cloak = cv2.bitwise_and(frame, frame, mask=inverse_mask)
        cloak_area_from_bg = cv2.bitwise_and(background, background, mask=mask)

        final = cv2.addWeighted(current_no_cloak, 1, cloak_area_from_bg, 1, 0)

        cv2.putText(final, "Press 'b' to recapture background | 'q' to quit",
                    (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240,240,240), 2)

        cv2.imshow("Invisibility Cloak", final)

        # ================= Write frame to video =================
        if record_video:
            out.write(final)
        # ========================================================

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('b'):
            print("📸 Re-capturing background...")
            background = get_background(cap)
            print("✅ Background updated.")

    cap.release()
    if record_video:
        out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
