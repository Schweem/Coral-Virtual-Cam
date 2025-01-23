import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter, load_delegate
import os
from collections import deque

# Paths for the object detection model
MODEL_PATH = os.path.join('models', 'ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite')
LABEL_PATH = os.path.join('models', 'coco_labels.txt')
EDGETPU_SHARED_LIB = '../tpulib/libedgetpu.1.dylib'

# Load the Edge TPU model
try:
    interpreter = Interpreter(
        model_path=MODEL_PATH,
        experimental_delegates=[load_delegate(EDGETPU_SHARED_LIB)]
    )
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("TPU initialized successfully")
except Exception as e:
    print(f"Error loading TPU: {e}")
    exit()

def load_labels(label_path):
    """Load labels from a file."""
    with open(label_path, 'r') as f:
        lines = f.readlines()
    labels = {i: line.strip() for i, line in enumerate(lines)}
    return labels

# Load the labels
try:
    labels = load_labels(LABEL_PATH)
except Exception as e:
    print(f"Error loading labels: {e}")
    labels = {}

def process_frame_tpu(frame):
    """Process frame through TPU for object detection"""
    input_shape = input_details[0]['shape']
    processed = cv2.resize(frame, (input_shape[2], input_shape[1]))

    input_data = np.expand_dims(processed, axis=0)
    if input_details[0]['dtype'] == np.uint8:
        input_data = input_data.astype(np.uint8)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    output_data = {
        'boxes': interpreter.get_tensor(output_details[0]['index'])[0],
        'classes': interpreter.get_tensor(output_details[1]['index'])[0],
        'scores': interpreter.get_tensor(output_details[2]['index'])[0],
    }
    return output_data

def apply_motion_extraction(current_frame, delayed_frame, blend_alpha):
    """
    Apply traditional motion extraction by inverting both frames
    and blending them with a slight temporal offset.
    """
    # Invert both the current frame and the delayed frame
    inverted_current = cv2.bitwise_not(current_frame)
    inverted_delayed = cv2.bitwise_not(delayed_frame)

    # Blend the inverted frames
    motion_highlight = cv2.addWeighted(
        inverted_current, blend_alpha, inverted_delayed, blend_alpha, 0
    )

    return motion_highlight

def apply_channel_offsets(frame, offsets):
    """Apply channel offsets to the frame."""
    b, g, r = cv2.split(frame)
    b = np.roll(b, offsets[0], axis=1)
    g = np.roll(g, offsets[1], axis=1)
    r = np.roll(r, offsets[2], axis=1)
    return cv2.merge((b, g, r))

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Parameters
frame_buffer_size = 15
frame_buffer = deque(maxlen=frame_buffer_size)
blend_alpha = 0.6
channel_offsets = [0, 0, 0]
effect_mode = 0
score_threshold = 0.5  # Minimum confidence for object detection

print("\nControls:")
print("q - Quit")
print("m - Toggle RGB mode")
print("a/s - Decrease/Increase frame offset")
print("z/x - Decrease/Increase blend")
print("r/t - Adjust Red channel")
print("g/h - Adjust Green channel")
print("b/n - Adjust Blue channel")
print("space - Reset all effects\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_buffer.append(frame)

    if len(frame_buffer) >= frame_buffer_size:
        delayed_frame = frame_buffer[0]

        # Apply traditional motion extraction with inverted frames
        motion_highlight = apply_motion_extraction(frame, delayed_frame, blend_alpha)

        # Apply channel offsets if in RGB mode
        if effect_mode == 1:
            motion_highlight = apply_channel_offsets(motion_highlight, channel_offsets)

        # Object detection
        detection_results = process_frame_tpu(frame)
        height, width, _ = frame.shape
        for i, score in enumerate(detection_results['scores']):
            if score > score_threshold:
                ymin, xmin, ymax, xmax = detection_results['boxes'][i]
                x0, y0, x1, y1 = int(xmin * width), int(ymin * height), int(xmax * width), int(ymax * height)
                class_id = int(detection_results['classes'][i])
                label = labels.get(class_id, f"Class {class_id}")

                # Draw detection bounding boxes and labels
                cv2.rectangle(motion_highlight, (x0, y0), (x1, y1), (0, 255, 0), 2)
                cv2.putText(
                    motion_highlight,
                    f"{label}: {score:.2f}",
                    (x0, y0 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )

        # Display the result
        cv2.imshow('Motion + Object Detection', motion_highlight)

    # Handle keypresses
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('m'):
        effect_mode = 1 - effect_mode
    elif key == ord('a'):
        frame_buffer_size = max(1, frame_buffer_size - 1)
        frame_buffer = deque(frame_buffer, maxlen=frame_buffer_size)
    elif key == ord('s'):
        frame_buffer_size = min(30, frame_buffer_size + 1)
        frame_buffer = deque(frame_buffer, maxlen=frame_buffer_size)
    elif key == ord('z'):
        blend_alpha = max(0.1, blend_alpha - 0.1)
    elif key == ord('x'):
        blend_alpha = min(1.0, blend_alpha + 0.1)
    elif key == ord('r'):
        channel_offsets[2] = max(-50, channel_offsets[2] - 1)
    elif key == ord('t'):
        channel_offsets[2] = min(50, channel_offsets[2] + 1)
    elif key == ord('g'):
        channel_offsets[1] = max(-50, channel_offsets[1] - 1)
    elif key == ord('h'):
        channel_offsets[1] = min(50, channel_offsets[1] + 1)
    elif key == ord('b'):
        channel_offsets[0] = max(-50, channel_offsets[0] - 1)
    elif key == ord('n'):
        channel_offsets[0] = min(50, channel_offsets[0] + 1)
    elif key == ord(' '):
        channel_offsets = [0, 0, 0]
        blend_alpha = 0.5
        frame_buffer_size = 15
        frame_buffer = deque(maxlen=frame_buffer_size)

# Cleanup
cap.release()
cv2.destroyAllWindows()