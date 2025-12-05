# untuk ngambil data gambar
import os
import uuid
import time
import cv2


IMAGES_PATH = os.path.join('Data')  # folder where images will be stored
NUM_IMAGES = 20                       # how many images to capture
DELAY = 0.5                           # seconds between captures
CAM_INDEX = 0                         # try 0, 1, or 2 if webcam fails


# Create directory if it doesn't exist
os.makedirs(IMAGES_PATH, exist_ok=True)

# Open webcam
cap = cv2.VideoCapture(CAM_INDEX)

if not cap.isOpened():
    raise Exception("❌ ERROR: Could not open webcam. Try CAM_INDEX = 1 or 2.")

print("✅ Webcam opened successfully.")
print("Press 'q' to quit manually.\n")

# Capture loop
for imgnum in range(NUM_IMAGES):
    print(f"Collecting image {imgnum}")

    ret, frame = cap.read()

    # If capture failed
    if not ret or frame is None:
        print("❌ Failed to capture frame. Retrying...")
        time.sleep(0.2)
        continue

    img_name = os.path.join(IMAGES_PATH, f"{uuid.uuid1()}.jpg")
    cv2.imwrite(img_name, frame)

    # Show live preview
    cv2.imshow("Webcam - Press Q to quit", frame)

    # Delay between captures
    time.sleep(DELAY)

    # Exit manually
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("🛑 Quit by user.")
        break

cap.release()
cv2.destroyAllWindows()

print("\n🎉 DONE! Images saved to:", IMAGES_PATH)
