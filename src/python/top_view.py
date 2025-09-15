import cv2
import numpy as np
import math

def equirectangular_to_topdown(equi_img, fov_deg=180, out_w=1024, out_h=1024):
    """
    Convert an equirectangular image to a top-down (nadir) view.
    
    equi_img : input equirectangular BGR image
    fov_deg  : field of view for the top-down crop (180 for full hemisphere)
    out_w, out_h : output top-down resolution
    """
    H, W = equi_img.shape[:2]

    # Nadir direction (looking straight down)
    center_lon = 0.0  # doesn't matter for nadir
    center_lat = -90.0  # -90 degrees latitude = straight down

    # Convert to radians
    lam0 = math.radians(center_lon)
    phi0 = math.radians(center_lat)

    # Output pixel grid
    xs = np.linspace(-1, 1, out_w)
    ys = np.linspace(-1, 1, out_h)
    X, Y = np.meshgrid(xs, ys)

    # Radius in normalized image plane
    R = np.sqrt(X**2 + Y**2)
    R = np.clip(R, 1e-9, 1.0)

    # Convert polar coords in output to spherical coords
    max_theta = math.radians(fov_deg) / 2.0
    theta = R * max_theta
    phi = math.pi/2 - theta  # polar angle from zenith

    # Azimuth relative to image center
    az = np.arctan2(Y, X)

    # Direction vector for each pixel
    Dx = np.sin(phi) * np.cos(az)
    Dy = np.cos(phi)
    Dz = np.sin(phi) * np.sin(az)

    # Rotate so nadir points down
    # For nadir, forward vector = (0, -1, 0)
    # Rotation: y->-z, z->y
    Dxn = Dx
    Dyn = -Dz
    Dzn = Dy

    # Convert to equirect coordinates
    lon = np.arctan2(Dzn, Dxn)
    lat = np.arcsin(np.clip(Dyn, -1, 1))

    map_x = ((lon + math.pi) / (2 * math.pi) * W).astype(np.float32)
    map_y = ((math.pi/2 - lat) / math.pi * H).astype(np.float32)

    topdown_img = cv2.remap(equi_img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    return topdown_img


if __name__ == "__main__":
    # Example usage
    filename = "live_out\equi_sample_2.jpg"  # Replace with your equirectangular image
    equi = cv2.imread(filename)
    topdown = equirectangular_to_topdown(equi, fov_deg=180, out_w=1024, out_h=1024)
    cv2.imwrite("live_out\equi_sample_2_topdown.jpg", topdown)
    cv2.imshow("Top-Down View", topdown)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
