import cv2
import numpy as np
import math
import time

# Open webcam
cam = cv2.VideoCapture(0)

# Background subtractor
bg_sub = cv2.createBackgroundSubtractorMOG2()

prevX = None
prevY = None
direction = "Detecting..."
distanceCM = 0
distanceInch = 0

THRESH = 15          # Movement threshold
PIX_TO_CM = 0.026    # Pixel to cm conversion
PAUSE = 1.0          # 1 second pause
lastUpdate = time.time()

while True:
    ret, frame = cam.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    mask = bg_sub.apply(gray)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        big = max(contours, key=cv2.contourArea)
        if cv2.contourArea(big) > 1500:
            x, y, w, h = cv2.boundingRect(big)
            cx = x + w // 2
            cy = y + h // 2

            # Draw green box and red center dot
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
            cv2.circle(frame, (cx, cy), 5, (0,0,255), -1)

            if prevX is not None and (time.time() - lastUpdate) > PAUSE:
                dx = cx - prevX
                dy = cy - prevY

                # Distance calculation
                distPx = math.sqrt(dx*dx + dy*dy)
                distanceCM = distPx * PIX_TO_CM
                distanceInch = distanceCM / 2.54

                # Direction logic
                if abs(dx) > abs(dy):
                    if dx > THRESH:
                        direction = "RIGHT"
                    elif dx < -THRESH:
                        direction = "LEFT"
                else:
                    if dy < -THRESH:
                        direction = "FORWARD"
                    elif dy > THRESH:
                        direction = "BACKWARD"

                lastUpdate = time.time()

            prevX, prevY = cx, cy

    # Show direction and distance on screen
    cv2.putText(frame, "Direction: " + direction, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
    cv2.putText(frame, f"Distance: {distanceCM:.2f} cm / {distanceInch:.2f} inch",
                (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

    cv2.imshow("Movement Detection", frame)

    if cv2.waitKey(30) & 0xFF == 27:
        break

cam.release()
cv2.destroyAllWindows()
