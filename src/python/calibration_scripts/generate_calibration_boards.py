# make_a3_hybrid_aruco.py
import math
import numpy as np
import cv2
from PIL import Image

# ================= SETTINGS =================
DPI = 300
MARGIN_MM = 10                 # white margin around page
CHARUCO_DICT = cv2.aruco.DICT_5X5_1000

# A3 size (portrait)
W_MM, H_MM = 297.0, 420.0

# Marker sizes
BIG_MARKER_MM = 90.0           # large corner markers (≈9 cm)
EDGE_MARKER_MM = 60.0          # medium markers along edges (≈6 cm)
EDGE_OFFSET_MM = 30.0          # distance of edge markers from paper edge
# ============================================

def mm_to_px(mm, dpi): 
    return int(round((mm / 25.4) * dpi))

PAGE_W_PX = mm_to_px(W_MM, DPI)
PAGE_H_PX = mm_to_px(H_MM, DPI)
MARGIN_PX = mm_to_px(MARGIN_MM, DPI)

aruco_dict = cv2.aruco.getPredefinedDictionary(CHARUCO_DICT)

def draw_marker(marker_id, size_px):
    return cv2.aruco.generateImageMarker(aruco_dict, marker_id, size_px)

def build_hybrid_board():
    # Start with white page
    page = np.full((PAGE_H_PX, PAGE_W_PX), 255, np.uint8)

    # ---- Place 4 large corner markers ----
    big_px = mm_to_px(BIG_MARKER_MM, DPI)
    corners = [
        (MARGIN_PX, MARGIN_PX),  # top-left
        (PAGE_W_PX - MARGIN_PX - big_px, MARGIN_PX),  # top-right
        (MARGIN_PX, PAGE_H_PX - MARGIN_PX - big_px),  # bottom-left
        (PAGE_W_PX - MARGIN_PX - big_px, PAGE_H_PX - MARGIN_PX - big_px)  # bottom-right
    ]
    for i, (x, y) in enumerate(corners):
        marker = draw_marker(i, big_px)
        page[y:y+big_px, x:x+big_px] = marker

    # ---- Place medium edge markers ----
    edge_px = mm_to_px(EDGE_MARKER_MM, DPI)
    offset_px = mm_to_px(EDGE_OFFSET_MM, DPI)

    # Top edge (centered)
    x_top = (PAGE_W_PX - edge_px) // 2
    y_top = MARGIN_PX + offset_px
    page[y_top:y_top+edge_px, x_top:x_top+edge_px] = draw_marker(10, edge_px)

    # Bottom edge
    y_bottom = PAGE_H_PX - MARGIN_PX - offset_px - edge_px
    page[y_bottom:y_bottom+edge_px, x_top:x_top+edge_px] = draw_marker(11, edge_px)

    # Left edge
    x_left = MARGIN_PX + offset_px
    y_left = (PAGE_H_PX - edge_px) // 2
    page[y_left:y_left+edge_px, x_left:x_left+edge_px] = draw_marker(12, edge_px)

    # Right edge
    x_right = PAGE_W_PX - MARGIN_PX - offset_px - edge_px
    page[y_left:y_left+edge_px, x_right:x_right+edge_px] = draw_marker(13, edge_px)

    # Save final image
    Image.fromarray(page).save("aruco_hybrid_a3.png", dpi=(DPI, DPI))
    print("[Hybrid A3] Saved → aruco_hybrid_a3.png")

# if __name__ == "__main__":
#     build_hybrid_board()



##################################################################

# make_letter_charuco_aruco.py
import math
import numpy as np
import cv2
from PIL import Image

# =============== USER SETTINGS =================
ORIENTATION = "portrait"       # "portrait" (8.5x11) or "landscape" (11x8.5)
DPI = 300                      # 300 dpi = crisp print
MARGIN_MM = 10                 # white margin around board on the page

#### letter size (8.5x11 inch) ####
# # --- ChArUCo (chess squares w/ ArUco inside) ---
# SQUARE_MM = 30.0               # chess square edge length (mm) — change to taste
# MARKER_MM = 0.70 * SQUARE_MM   # inner marker size (mm) ~70% of square
# CHARUCO_DICT = cv2.aruco.DICT_5X5_1000

# # --- ArUco GridBoard (markers only) ---
# GB_MARKER_MM = 36.0            # marker edge length (mm)
# GB_GAP_MM    = 6.0             # gap between markers (mm)
# GRID_DICT = cv2.aruco.DICT_5X5_1000


#### A3 size (297x420 mm) ####
# --- ChArUCo (chess squares w/ ArUco inside) ---
SQUARE_MM = 90.0               # larger chess square edge
MARKER_MM = 0.70 * SQUARE_MM   # ~70% of square
CHARUCO_DICT = cv2.aruco.DICT_5X5_1000

# --- ArUco GridBoard (markers only) ---
GB_MARKER_MM = 90.0            # very large markers (9 cm each)
GB_GAP_MM    = 12.0            # ~1.2 cm gap between markers
GRID_DICT = cv2.aruco.DICT_5X5_1000

# ===============================================

# Letter in mm
# W_MM, H_MM = (215.9, 279.4) if ORIENTATION == "portrait" else (279.4, 215.9)

# # Output filenames
# OUT_CHARUCO = "charuco_letter.png"
# OUT_GRID    = "aruco_grid_letter.png"

# ##########################

# A3 in mm
W_MM, H_MM = (297.0, 420.0) if ORIENTATION == "portrait" else (420.0, 297.0)

# Update output filenames for clarity
OUT_CHARUCO = "charuco_a3.png"
OUT_GRID    = "aruco_grid_a3.png"

# ###########################


def mm_to_px(mm, dpi): return int(round((mm / 25.4) * dpi))

PAGE_W_PX = mm_to_px(W_MM, DPI)
PAGE_H_PX = mm_to_px(H_MM, DPI)
MARGIN_PX = mm_to_px(MARGIN_MM, DPI)

def build_charuco():
    usable_w_mm = W_MM - 2*MARGIN_MM
    usable_h_mm = H_MM - 2*MARGIN_MM

    squaresX = int(math.floor(usable_w_mm / SQUARE_MM))
    squaresY = int(math.floor(usable_h_mm / SQUARE_MM))

    board_w_mm = squaresX * SQUARE_MM
    board_h_mm = squaresY * SQUARE_MM

    board_w_px = mm_to_px(board_w_mm, DPI)
    board_h_px = mm_to_px(board_h_mm, DPI)

    aruco_dict = cv2.aruco.getPredefinedDictionary(CHARUCO_DICT)
    board = cv2.aruco.CharucoBoard(
        (squaresX, squaresY),
        SQUARE_MM / 1000.0,        # meters (used later for pose)
        MARKER_MM / 1000.0,
        aruco_dict
    )

    img = board.generateImage((board_w_px, board_h_px), marginSize=0, borderBits=1)

    page = np.full((PAGE_H_PX, PAGE_W_PX), 255, np.uint8)
    off_x = MARGIN_PX + (mm_to_px(usable_w_mm, DPI) - board_w_px) // 2
    off_y = MARGIN_PX + (mm_to_px(usable_h_mm, DPI) - board_h_px) // 2
    page[off_y:off_y+board_h_px, off_x:off_x+board_w_px] = img

    Image.fromarray(page).save(OUT_CHARUCO, dpi=(DPI, DPI))
    print(f"[ChArUCo] {squaresX}x{squaresY} squares @ {SQUARE_MM:.1f}mm "
          f"(marker {MARKER_MM:.1f}mm) → {OUT_CHARUCO}")

# def build_gridboard():
#     usable_w_mm = W_MM - 2*MARGIN_MM
#     usable_h_mm = H_MM - 2*MARGIN_MM
#     pitch_mm = GB_MARKER_MM + GB_GAP_MM

#     markersX = int(math.floor((usable_w_mm + GB_GAP_MM) / pitch_mm))
#     markersY = int(math.floor((usable_h_mm + GB_GAP_MM) / pitch_mm))

#     board_w_mm = markersX * GB_MARKER_MM + (markersX - 1) * GB_GAP_MM
#     board_h_mm = markersY * GB_MARKER_MM + (markersY - 1) * GB_GAP_MM

#     board_w_px = mm_to_px(board_w_mm, DPI)
#     board_h_px = mm_to_px(board_h_mm, DPI)

#     aruco_dict = cv2.aruco.getPredefinedDictionary(GRID_DICT)
#     board = cv2.aruco.GridBoard(
#         (markersX, markersY),
#         GB_MARKER_MM / 1000.0,
#         GB_GAP_MM / 1000.0,
#         aruco_dict
#     )

#     img = board.generateImage((board_w_px, board_h_px), marginSize=0, borderBits=1)
#     page = np.full((PAGE_H_PX, PAGE_W_PX), 255, np.uint8)
#     off_x = MARGIN_PX + (mm_to_px(usable_w_mm, DPI) - board_w_px) // 2
#     off_y = MARGIN_PX + (mm_to_px(usable_h_mm, DPI) - board_h_px) // 2
#     page[off_y:off_y+board_h_px, off_x:off_x+board_w_px] = img

#     Image.fromarray(page).save(OUT_GRID, dpi=(DPI, DPI))
#     print(f"[GridBoard] {markersX}x{markersY} markers @ {GB_MARKER_MM:.1f}mm "
#           f"(gap {GB_GAP_MM:.1f}mm) → {OUT_GRID}")

if __name__ == "__main__":
    build_charuco()
#     build_gridboard()
    print(f"Page: {W_MM:.1f}×{H_MM:.1f} mm @ {DPI} dpi → {PAGE_W_PX}×{PAGE_H_PX} px")
