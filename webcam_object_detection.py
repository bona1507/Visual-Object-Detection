import cv2
from YOLOv6 import YOLOv6
import subprocess

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Initialize YOLOv6 object detector
model_path = "" #change this with spesific model_path
yolov6_detector = YOLOv6(model_path, conf_thres=0.7, iou_thres=0.5)

cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)
previous_class_id = None

while cap.isOpened():

    # Read frame from the video
    ret, frame = cap.read()

    if not ret:
        break

    class_names = ['apel', 'botol', 'bunga', 'bus_kota', 'cincin', 'gedung', 'gelas', 'helm', 'jam_dinding', 'jam_tangan', 'jendela', 'kacamata', 'kalung', 'kapal_penumpang', 'laki_laki', 'lampu_gantung', 'lampu_jalan', 'mobil', 'patung', 'payung', 'perahu', 'perempuan', 'pohon', 'sendok', 'sepatu', 'sepeda', 'smartphone', 'topi']

    # Update object localizer
    boxes, scores, class_ids = yolov6_detector(frame)

    for class_id in class_ids:
        if class_id != previous_class_id and class_id is not None:
            print(class_names[class_id])
            previous_class_id = class_id
            command = "wsl espeak-ng -v mb-en1 '" + class_names[class_id] + "'"
            subprocess.run(command, shell=True)

    combined_img = yolov6_detector.draw_detections(frame)
    cv2.imshow("Detected Objects", combined_img)

    # Press key q to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()