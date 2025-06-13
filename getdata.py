import socket
import struct

UDP_IP = "127.0.0.1"
UDP_PORT = 4242

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

print("Listening for OpenTrack UDP packets (double precision)...")

while True:
    data, _ = sock.recvfrom(1024)
    x, y, z, yaw, pitch, roll = struct.unpack('<6d', data[:48])
    print(f"Yaw: {yaw:.0f}, Pitch: {pitch:.0f}, Roll: {roll:.0f} | X: {x:.0f}, Y: {y:.0f}, Z: {z:.0f}")
