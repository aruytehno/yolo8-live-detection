import cv2
from ultralytics import YOLO
from datetime import datetime
import os

# Загрузка модели
model = YOLO("yolov8n.pt")  # Модель автоматически загрузится

# Подготовка папки для скриншотов
os.makedirs("screenshots", exist_ok=True)

# Запуск веб-камеры
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Не удалось открыть камеру")
    exit()

print("✅ Камера запущена. Нажми Q — выход, S — сохранить кадр")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Детекция
    results = model(frame, verbose=False)
    annotated_frame = results[0].plot()

    # Показ окна
    cv2.imshow("YOLOv8n - Webcam", annotated_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("s"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"screenshots/frame_{timestamp}.jpg"
        cv2.imwrite(filename, annotated_frame)
        print(f"📸 Сохранено: {filename}")

cap.release()
cv2.destroyAllWindows()
