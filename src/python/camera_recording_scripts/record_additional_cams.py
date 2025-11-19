import cv2
import threading
import os, time, signal, sys

# === Base directory ===
SAVE_DIR = r"C:\Users\sures\Documents\Children_school_data"

# === Get session name from command line ===
if len(sys.argv) < 2:
    print("❌ Usage: python record_cam.py <session_name>")
    sys.exit(1)

session = sys.argv[1].strip().replace(" ", "_")
session_path = os.path.join(SAVE_DIR, session)

if os.path.exists(session_path):
    # add a modifier to avoid overwriting
    modifier = 1
    while True:
        new_session_path = f"{session_path}_{modifier}"
        if not os.path.exists(new_session_path):
            session_path = new_session_path
            break
        modifier += 1
os.makedirs(session_path, exist_ok=True)

print("🎬 ================================================")
print(f"   Multi-Camera Recording - Session: {session}")
print("===============================================\n")
print(f"📂 Saving recordings to:\n   {session_path}\n")

# === Cameras to skip ===
SKIP = [0, 4]  # webcam and OBS virtual device

# === Capture settings ===
FRAME_WIDTH  = 1920
FRAME_HEIGHT = 1080
FPS          = 30.0
FOURCC       = cv2.VideoWriter_fourcc(*'mp4v')  # 'mp4v' for .mp4 output

stop_flag = [False]

def handle_sigint(sig, frame):
    print("\n⚠️ Ctrl+C detected — stopping safely...")
    stop_flag[0] = True
signal.signal(signal.SIGINT, handle_sigint)

# === Discover connected cameras ===
cams = []
print("🔍 Scanning connected cameras...")
for i in range(10):
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)
    ret, frame = cap.read()
    if ret:
        print(f"✅ Camera {i} ready ({frame.shape[1]}x{frame.shape[0]})")
        cams.append(i)
    cap.release()

cams = [c for c in cams if c not in SKIP]
if not cams:
    print("❌ No usable cameras found.")
    sys.exit(1)

print(f"🎥 Recording from cameras: {cams}\n")

# === Recording thread function ===
def record_camera(idx):
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)

    ret, frame = cap.read()
    if not ret or frame is None:
        print(f"❌ Camera {idx} failed to start.")
        return

    h, w = frame.shape[:2]
    filename = f"camera_{idx}_{w}x{h}.mp4"
    path = os.path.join(session_path, filename)

    writer = cv2.VideoWriter(path, FOURCC, FPS, (w, h))
    if not writer.isOpened():
        print(f"❌ Could not open writer for camera {idx}")
        return

    print(f"▶️  Recording camera {idx} -> {filename}")

    while not stop_flag[0]:
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.02)
            continue
        writer.write(frame)
        cv2.imshow(f"Cam {idx}", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_flag[0] = True
            break

    cap.release()
    writer.release()
    print(f"💾 Saved camera {idx} -> {path}")

# === Launch recording threads ===
threads = []
for c in cams:
    t = threading.Thread(target=record_camera, args=(c,))
    t.start()
    threads.append(t)

print("🚀 Recording started!")
print("👉 Press ENTER in the console or 'q' in any window to stop.\n")

input()
stop_flag[0] = True

for t in threads:
    t.join()

cv2.destroyAllWindows()
print("\n✅ All recordings completed successfully.")
print(f"📁 Files saved in: {session_path}")
