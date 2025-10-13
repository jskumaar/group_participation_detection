import subprocess
import re
import os

# === Step 1: Get device list ===
result = subprocess.run(
    ["ffmpeg", "-hide_banner", "-list_devices", "true", "-f", "dshow", "-i", "dummy"],
    stderr=subprocess.PIPE,
    text=True
)
output = result.stderr

# === Step 2: Extract all "Luxonis UVC Camera" alternative names ===
pattern = r'Alternative name\s+"(@device_pnp_.*?)"'
matches = re.findall(pattern, output)

if not matches:
    print("❌ No Luxonis cameras found!")
    exit()

os.makedirs(r"C:\1", exist_ok=True)

# === Step 3: Build FFmpeg command ===
cmd = ["ffmpeg", "-y"]
for alt_name in matches:
    cmd += [
        "-f", "dshow",
        "-rtbufsize", "1M",
        "-r", "30",
        "-i", f"video={alt_name}"
    ]

# === Step 4: Add output mappings (one file per camera) ===
for i in range(len(matches)):
    cmd += [
        "-map", str(i),
        "-c:v", "libx264",
        "-preset", "ultrafast",
        f"C:\\1\\luxonis_cam{i+1}.mp4"
    ]

print("\n🚀 Running FFmpeg to record all Luxonis cameras:")
print(" ".join(cmd))
subprocess.run(cmd)
