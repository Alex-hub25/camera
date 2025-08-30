#!/usr/bin/env python3
"""
PaKit - Box dimension + delivery-data extractor for Raspberry Pi + Arducam 12MP

Requirements:
 - OpenCV (cv2)
 - numpy
 - pytesseract
 - imutils

Place an ArUco marker of known physical width (in mm) in the same plane as the box.
Adjust CAMERA_INDEX and marker_size_mm to your setup.
"""

import cv2
import numpy as np
import pytesseract
import re
import json
import time
import os
# from imutils import resize

# ====== User settings ======
CAMERA_INDEX = 0  # Change if camera uses another index or capture method
ARUCO_DICT = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
ARUCO_PARAMS = cv2.aruco.DetectorParameters_create()
marker_size_mm = 50.0   # physical width of printed ArUco marker in millimeters; CHANGE to your printed marker size
OUTPUT_JSON = "last_reading.json"
CAPTURE_W = 4056  # target capture resolution (Arducam 12MP: use appropriate capture/resolution)
CAPTURE_H = 3040
# Optionally downscale for faster processing
DOWNSCALE_TO = 1600

# OCR config
TESSERACT_CONFIG = r'--oem 1 --psm 6'  # psm 6 = assume a single uniform block of text

# Regex patterns for parsing
EMAIL_RE = re.compile(r'[\w\.-]+@[\w\.-]+\.\w+')
PHONE_RE = re.compile(r'(\+?\d[\d\-\s\(\)]{5,}\d)')  # basic phone finder
ZIP_RE = re.compile(r'\b\d{5}(?:-\d{4})?\b')  # US zip format; change for other countries
AMOUNT_RE = re.compile(r'(\$?\s?\d{1,3}(?:[,\s]\d{3})*(?:\.\d{2})?)')  # $123.45 or 1,234.56
CITY_COUNTRY_RE = re.compile(r'\b([A-Za-z\-\s]{2,50})[,]\s*([A-Za-z\-\s]{2,50})\b')  # naive city, country
# ============================

def capture_frame():
    """Capture one high-res frame from the camera."""
    # Try VideoCapture; adapt if using Arducam SDK
    cap = cv2.VideoCapture(CAMERA_INDEX)
    # Set resolution if supported
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_H)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera. Check camera index/driver.")
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("Failed to capture image from camera.")
    return frame

def find_aruco_marker(frame_gray):
    """Detect ArUco markers and return the largest marker's pixel width and its corner coordinates."""
    corners, ids, rejected = cv2.aruco.detectMarkers(frame_gray, ARUCO_DICT, parameters=ARUCO_PARAMS)
    if ids is None or len(corners) == 0:
        return None
    # pick largest marker by bounding box area
    best = None
    best_area = 0
    for c in corners:
        c_arr = c.reshape((4,2))
        (x_min, y_min) = c_arr.min(axis=0)
        (x_max, y_max) = c_arr.max(axis=0)
        area = (x_max - x_min) * (y_max - y_min)
        if area > best_area:
            best_area = area
            best = c_arr
    # compute pixel width as average of top and bottom edge lengths
    top = np.linalg.norm(best[0] - best[1])
    bottom = np.linalg.norm(best[2] - best[3])
    pixel_width = (top + bottom) / 2.0
    return {"corners": best, "pixel_width": pixel_width, "area": best_area}

def compute_pixel_per_mm(marker_info):
    """Return pixel/mm given marker_info and known marker_size_mm."""
    if marker_info is None:
        return None
    return marker_info["pixel_width"] / marker_size_mm

def detect_box_contour(frame_gray):
    """Detect the largest rectangular-like contour in the frame (assumed to be the box)."""
    blurred = cv2.GaussianBlur(frame_gray, (5,5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    # close gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    # pick largest by area
    c = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    # return the bounding rotated rect for measurement
    rect = cv2.minAreaRect(c)  # ((cx,cy),(w,h),angle)
    box_points = np.array(cv2.boxPoints(rect), dtype="int")
    return {"contour": c, "rect": rect, "box_points": box_points}

def measure_box(rect, pixel_per_mm):
    """From cv2.minAreaRect rect and pixel_per_mm, return size in mm (width, height) of that rectangle."""
    ((cx, cy), (w_px, h_px), angle) = rect
    # convert pixels to mm
    if pixel_per_mm is None or pixel_per_mm <= 0:
        return None
    w_mm = float(w_px) / pixel_per_mm
    h_mm = float(h_px) / pixel_per_mm
    return {"width_mm": w_mm, "height_mm": h_mm, "angle": angle, "center": (cx,cy)}

def extract_label_region(frame):
    """Run a coarse text detection to find label region(s). This is heuristic: finds largest text-like area via morphology on thresholded image."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # adaptive threshold to highlight dark text on light background
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 25, 15)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25,5))
    dilated = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    # pick contours that are wide and not tiny
    candidates = []
    h_img, w_img = gray.shape
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        if w < 0.15 * w_img or h < 0.03 * h_img:
            continue
        candidates.append((x,y,w,h))
    if not candidates:
        return None
    # pick the candidate with max area
    x,y,w,h = max(candidates, key=lambda r: r[2]*r[3])
    # pad a bit
    pad_x = int(w * 0.03)
    pad_y = int(h * 0.05)
    x0 = max(0, x - pad_x)
    y0 = max(0, y - pad_y)
    x1 = min(w_img, x + w + pad_x)
    y1 = min(h_img, y + h + pad_y)
    return frame[y0:y1, x0:x1].copy()

def ocr_text_from_image(img):
    """Preprocess and run pytesseract OCR; return full text."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # increase contrast
    gray = cv2.equalizeHist(gray)
    # optional threshold
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    text = pytesseract.image_to_string(th, config=TESSERACT_CONFIG)
    return text

def parse_delivery_fields(text):
    """Extract heuristics for name, address, city, zip, country, email, phone, order description, amount."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    joined = " ".join(lines)
    result = {
        "raw": text,
        "name": None,
        "address": None,
        "city": None,
        "zip": None,
        "country": None,
        "email": None,
        "phone": None,
        "order_description": None,
        "amount": None,
    }
    # email
    s = EMAIL_RE.search(joined)
    if s:
        result["email"] = s.group(0)
    # phone
    s = PHONE_RE.search(joined)
    if s:
        result["phone"] = s.group(0)
    # zip
    s = ZIP_RE.search(joined)
    if s:
        result["zip"] = s.group(0)
    # amount
    s = AMOUNT_RE.search(joined)
    if s:
        result["amount"] = s.group(1).strip()
    # naive name/address: assume first line is name, next 1-3 lines are address
    if len(lines) >= 1:
        result["name"] = lines[0]
    if len(lines) >= 2:
        # treat lines 1..3 as address block until a line contains zip or country guess
        addr_lines = []
        for ln in lines[1:5]:
            addr_lines.append(ln)
        result["address"] = ", ".join(addr_lines)
    # city/country attempt: look for pattern "City, Country" in text
    s = CITY_COUNTRY_RE.search(joined)
    if s:
        result["city"] = s.group(1).strip()
        result["country"] = s.group(2).strip()
    # improved heuristics could be added here for specific label formats
    # order description: try to find line containing words like "Item", "Desc", "Description", "SKU"
    desc_candidates = [ln for ln in lines if re.search(r'item|description|desc|sku|order', ln, re.I)]
    if desc_candidates:
        result["order_description"] = desc_candidates[0]
    else:
        # fallback: last non-empty line if it contains currency or product-like text
        if len(lines) >= 1:
            result["order_description"] = lines[-1]
    return result

def process_frame(frame):
    """Full pipeline on a single frame."""
    # optional downscale for speed
    h, w = frame.shape[:2]
    scale = 1.0
    if max(w,h) > DOWNSCALE_TO:
        scale = DOWNSCALE_TO / float(max(w,h))
        frame_proc = cv2.resize(frame, (int(w*scale), int(h*scale)))
    else:
        frame_proc = frame.copy()

    gray = cv2.cvtColor(frame_proc, cv2.COLOR_BGR2GRAY)

    # find ArUco marker
    marker = find_aruco_marker(gray)
    pixel_per_mm = compute_pixel_per_mm(marker) if marker else None

    box = detect_box_contour(gray)

    dimensions = None
    if box and pixel_per_mm:
        # note: rect returned is in downscaled coordinates -> convert measurements back to original if scaled
        rect = box["rect"]
        # if we scaled image, rect dims are in scaled pixels; so pixel_per_mm is based on scaled pixels as well
        measure = measure_box(rect, pixel_per_mm)
        dimensions = measure

    # detect label region and OCR
    label_img = extract_label_region(frame_proc)
    ocr_text = None
    parsed = None
    if label_img is not None:
        ocr_text = ocr_text_from_image(label_img)
        parsed = parse_delivery_fields(ocr_text)

    # compose results (convert mm back to original scaling if necessary)
    result = {
        "timestamp": time.time(),
        "marker_found": marker is not None,
        "pixel_per_mm": float(pixel_per_mm) if pixel_per_mm else None,
        "dimensions_mm": dimensions,
        "ocr_raw": ocr_text,
        "delivery_fields": parsed
    }
    return result, frame_proc, marker, box, label_img

def main_loop(save_json=True, single_shot=True):
    frame = capture_frame()
    res, frame_proc, marker, box, label_img = process_frame(frame)

    # show some debugging overlay (optional)
    vis = frame_proc.copy()
    if marker is not None:
        pts = marker["corners"].astype(int)
        cv2.polylines(vis, [pts], True, (0,255,0), 2)
        cv2.putText(vis, f"Marker px width: {marker['pixel_width']:.1f}", (pts[0][0]+5, pts[0][1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    if box is not None:
        cv2.polylines(vis, [box["box_points"]], True, (255,0,0), 2)
        if res["dimensions_mm"]:
            wmm = res["dimensions_mm"]["width_mm"]
            hmm = res["dimensions_mm"]["height_mm"]
            cx,cy = res["dimensions_mm"]["center"]
            cv2.putText(vis, f"{wmm:.1f}mm x {hmm:.1f}mm", (int(cx), int(cy)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)
    if label_img is not None:
        # save crop for debugging
        cv2.imwrite("last_label_crop.jpg", label_img)

    # print results
    print("=== PaKit Reading ===")
    print(json.dumps({
        "timestamp": time.ctime(res["timestamp"]),
        "marker_found": res["marker_found"],
        "pixel_per_mm": res["pixel_per_mm"],
        "dimensions_mm": res["dimensions_mm"],
        "delivery_fields": res["delivery_fields"]
    }, indent=2, default=str))

    if save_json:
        out = {
            "human_timestamp": time.ctime(res["timestamp"]),
            "marker_found": res["marker_found"],
            "pixel_per_mm": res["pixel_per_mm"],
            "dimensions_mm": res["dimensions_mm"],
            "delivery_fields": res["delivery_fields"],
            "ocr_raw": res["ocr_raw"]
        }
        with open(OUTPUT_JSON, "w") as f:
            json.dump(out, f, indent=2, default=str)
        print(f"Saved JSON to {OUTPUT_JSON}")

    # optionally show visualization (only if interactive)
    # cv2.imshow("vis", vis)
    # cv2.waitKey(0)
    return res

if __name__ == "__main__":
    try:
        result = main_loop(single_shot=True)
    except Exception as e:
        print("Error:", e)
        raise
