import cv2
import numpy as np
from pycoral.utils import edgetpu
from pycoral.adapters import detect, common
from pycoral.utils.dataset import read_label_file
from tflite_runtime.interpreter import Interpreter, load_delegate
import os

# Initialize Edge TPU with COCO model
MODEL_PATH = os.path.join('models', 'ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite')
LABEL_PATH = os.path.join('models', 'coco_labels.txt')
EDGETPU_SHARED_LIB = '../tpulib/libedgetpu.1.dylib'  # Path to the Edge TPU library

# Load the Edge TPU model
try:
    interpreter = Interpreter(
        model_path=MODEL_PATH,
        experimental_delegates=[load_delegate(EDGETPU_SHARED_LIB)]
    )
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_size = common.input_size(interpreter)
    print(f"Model loaded successfully with input size: {input_size}")
    labels = read_label_file(LABEL_PATH)
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

def process_frame(frame, score_threshold=0.4):
    """Process frame through Edge TPU and return detections"""
    # Resize frame to match model input size
    height, width = frame.shape[:2]
    input_shape = input_details[0]['shape']
    input_size = (input_shape[2], input_shape[1])
    resized_frame = cv2.resize(frame, input_size)
    
    # Prepare input tensor
    input_data = np.expand_dims(resized_frame, axis=0)
    # The model expects uint8 input
    if input_details[0]['dtype'] == np.uint8:
        input_data = input_data.astype(np.uint8)
    
    # Run inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    # Get outputs
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Bounding boxes
    class_ids = interpreter.get_tensor(output_details[1]['index'])[0]  # Class IDs
    scores = interpreter.get_tensor(output_details[2]['index'])[0]  # Confidence scores
    
    # Filter detections based on score threshold
    valid_detections = scores > score_threshold
    
    # Draw bounding boxes and labels
    result_frame = frame.copy()
    for i in range(len(scores)):
        if valid_detections[i]:
            # Get normalized coordinates
            ymin, xmin, ymax, xmax = boxes[i]
            
            # Convert to pixel coordinates
            x0 = int(xmin * width)
            y0 = int(ymin * height)
            x1 = int(xmax * width)
            y1 = int(ymax * height)
            
            # Draw rectangle and label
            cv2.rectangle(result_frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
            
            # Get class name and score
            class_id = int(class_ids[i])
            score = scores[i]
            label = f"{labels.get(class_id, class_id)} {score:.2f}"
            
            # Add label above the bounding box
            cv2.putText(result_frame, label, (x0, y0-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Create detection objects for motion tracking
    detections = []
    for i in range(len(scores)):
        if valid_detections[i]:
            class_id = int(class_ids[i])
            score = scores[i]
            ymin, xmin, ymax, xmax = boxes[i]
            detection = type('Detection', (), {
                'id': class_id,
                'score': score,
                'bbox': type('BBox', (), {
                    'xmin': xmin,
                    'ymin': ymin,
                    'xmax': xmax,
                    'ymax': ymax
                })
            })
            detections.append(detection)
    
    return result_frame, detections

# Open the webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'q' to exit.")
print("Press 's' to save a screenshot.")

frame_count = 0
detection_history = []
motion_threshold = 0.1  # Threshold for significant motion

while True:
    # Capture frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not capture video frame.")
        break
    
    frame_count += 1
    
    # Process frame through Edge TPU
    processed_frame, detections = process_frame(frame)
    
    # Track motion by comparing detection positions
    if detection_history:
        for curr_obj in detections:
            for prev_obj in detection_history[-1]:
                if prev_obj.id == curr_obj.id:
                    # Calculate movement
                    prev_center = ((prev_obj.bbox.xmin + prev_obj.bbox.xmax) / 2,
                                 (prev_obj.bbox.ymin + prev_obj.bbox.ymax) / 2)
                    curr_center = ((curr_obj.bbox.xmin + curr_obj.bbox.xmax) / 2,
                                 (curr_obj.bbox.ymin + curr_obj.bbox.ymax) / 2)
                    
                    movement = np.sqrt((curr_center[0] - prev_center[0])**2 +
                                     (curr_center[1] - prev_center[1])**2)
                    
                    if movement > motion_threshold:
                        # Draw motion arrow
                        start_point = (int(prev_center[0] * frame.shape[1]),
                                     int(prev_center[1] * frame.shape[0]))
                        end_point = (int(curr_center[0] * frame.shape[1]),
                                   int(curr_center[1] * frame.shape[0]))
                        cv2.arrowedLine(processed_frame, start_point, end_point,
                                      (0, 0, 255), 2)
    
    # Keep detection history for motion tracking
    detection_history.append(detections)
    if len(detection_history) > 2:  # Only keep last 2 frames
        detection_history.pop(0)
    
    # Display frame counter and FPS
    cv2.putText(processed_frame, f"Frame: {frame_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Display the result
    cv2.imshow('Edge TPU Motion Detection', processed_frame)
    
    # Handle key presses
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        # Save screenshot
        filename = f"motion_detection_frame_{frame_count}.jpg"
        cv2.imwrite(filename, processed_frame)
        print(f"Screenshot saved as {filename}")

# Cleanup
cap.release()
cv2.destroyAllWindows()