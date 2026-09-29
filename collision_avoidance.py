from ultralytics import YOLO
import cv2

# ==========================================
# NAVIGEN - COLLISION AVOIDANCE MODULE
# ==========================================

# Load YOLO model
model = YOLO("yolo11n.pt")

# Open laptop webcam
cap = cv2.VideoCapture(0)

while True:

    # Read camera frame
    ret, frame = cap.read()

    if not ret:
        print("Camera not detected")
        break

    # Get frame dimensions
    height, width, _ = frame.shape

    # Divide camera view into 3 zones
    left_boundary = width // 3
    right_boundary = 2 * width // 3

    # Detect objects using YOLO
    results = model(frame)

    # Default values
    action = "MOVE FORWARD"
    obstacle_position = "NONE"
    risk_level = "LOW"

    # Check every detected object
    for box in results[0].boxes:

        # Get bounding box coordinates
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

        # Find center of the detected object
        center_x = (x1 + x2) / 2

        # Find height of detected object
        box_height = y2 - y1

        # Determine obstacle position
        if center_x < left_boundary:
            obstacle_position = "LEFT"

        elif center_x > right_boundary:
            obstacle_position = "RIGHT"

        else:
            obstacle_position = "CENTER"

        # ==========================================
        # COLLISION RISK AND AVOIDANCE DECISION
        # ==========================================

        # HIGH RISK
        if box_height > height * 0.45:

            risk_level = "HIGH"

            if obstacle_position == "LEFT":
                action = "TURN RIGHT"

            elif obstacle_position == "RIGHT":
                action = "TURN LEFT"

            else:
                action = "STOP"

        # MEDIUM RISK
        elif box_height > height * 0.25:

            risk_level = "MEDIUM"

            if obstacle_position == "LEFT":
                action = "TURN RIGHT"

            elif obstacle_position == "RIGHT":
                action = "TURN LEFT"

            else:
                action = "SLOW DOWN"

    # ==========================================
    # DRAW LEFT / CENTER / RIGHT ZONES
    # ==========================================

    cv2.line(
        frame,
        (left_boundary, 0),
        (left_boundary, height),
        (255, 255, 255),
        2
    )

    cv2.line(
        frame,
        (right_boundary, 0),
        (right_boundary, height),
        (255, 255, 255),
        2
    )

    # Draw YOLO detection boxes
    annotated_frame = results[0].plot()

    # ==========================================
    # DISPLAY COLLISION INFORMATION
    # ==========================================

    cv2.putText(
        annotated_frame,
        "Obstacle: " + obstacle_position,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    cv2.putText(
        annotated_frame,
        "Action: " + action,
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    cv2.putText(
        annotated_frame,
        "Risk: " + risk_level,
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    # Show the result
    cv2.imshow(
        "NAVIGEN - Collision Avoidance",
        annotated_frame
    )

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# Release camera
cap.release()

# Close all windows
cv2.destroyAllWindows()