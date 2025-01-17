import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter, load_delegate
from pycoral.adapters import common, segment

# Paths to the model and labels
MODEL_PATH = '../models/deeplabv3_mnv2_pascal_quant_edgetpu.tflite'
EDGETPU_SHARED_LIB = 'tpulib/libedgetpu.1.dylib'  # Path to the Edge TPU library
LABELS = [
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

# Initialize the interpreter with the Edge TPU delegate
try:
    interpreter = Interpreter(
        model_path=MODEL_PATH,
        experimental_delegates=[load_delegate(EDGETPU_SHARED_LIB)]
    )
    interpreter.allocate_tensors()
    input_size = common.input_size(interpreter)
    print(f"Model loaded successfully with input size: {input_size}")
except Exception as e:
    print(f"Error loading model or delegate: {e}")
    exit()

# Function to create a colormap for visualizing segmentation results
def create_pascal_label_colormap():
    colormap = np.zeros((256, 3), dtype=int)
    indices = np.arange(256, dtype=int)
    for shift in reversed(range(8)):
        for channel in range(3):
            colormap[:, channel] |= ((indices >> channel) & 1) << shift
        indices >>= 3
    return colormap

# Function to apply the colormap to the segmentation mask
def label_to_color_image(label):
    colormap = create_pascal_label_colormap()
    return colormap[label]

# Open the webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'q' to exit.")

while True:
    # Capture a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture video frame.")
        break

    # Resize the frame to the model's input size
    resized_frame = cv2.resize(frame, input_size)
    common.set_input(interpreter, resized_frame)

    # Run inference
    interpreter.invoke()

    # Get the segmentation result
    result = segment.get_output(interpreter)
    if len(result.shape) == 3:
        result = np.argmax(result, axis=-1)

    # Resize the result to match the original frame size
    result = cv2.resize(result, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Apply the colormap
    colorized_result = label_to_color_image(result).astype(np.uint8)

    # Blend the original frame with the colorized segmentation
    blended_frame = cv2.addWeighted(frame, 0.5, colorized_result, 0.5, 0)

    # Display the result
    cv2.imshow('Real-Time Semantic Segmentation', blended_frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()