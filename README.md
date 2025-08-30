# camera
computer vision
# system libs (Debian/Ubuntu)
sudo apt update
sudo apt install -y libatlas-base-dev libjpeg-dev libtiff5-dev libavcodec-dev libavformat-dev libswscale-dev \
    libv4l-dev libxvidcore-dev libx264-dev libgtk2.0-dev libcanberra-gtk* libqtgui4 libqt4-test \
    tesseract-ocr tesseract-ocr-eng

# pip
python3 -m pip install --upgrade pip
python3 -m pip install opencv-python-headless numpy imutils pytesseract pyserial
# NOTE: If using camera driver that exposes V4L2/video0 you can use cv2.VideoCapture(0).
# If Arducam provides different capture interface you might need Arducam SDK packages.
