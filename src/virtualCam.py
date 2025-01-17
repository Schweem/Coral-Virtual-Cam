import cv2
import numpy as np
import pyvirtualcam
from pyvirtualcam import PixelFormat
from tflite_runtime.interpreter import Interpreter, load_delegate
from pycoral.adapters import common, segment

# Paths to the model and tpulib
MODEL_PATH = '../models/deeplabv3_mnv2_pascal_quant_edgetpu.tflite'
EDGETPU_SHARED_LIB = 'tpulib/libedgetpu.1.dylib'  # Path to the Edge TPU library

# Load the Edge TPU model
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

# Open the webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Initialize virtual camera
with pyvirtualcam.Camera(width=640, height=480, fps=40, fmt=PixelFormat.BGR) as cam:
    print(f'Using virtual camera: {cam.device}')

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

        # Create a binary mask where the 'person' class (index 15) is detected
        mask = (result == 15).astype(np.uint8) * 255

        # Resize the mask to match the original frame size
        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Apply the mask to the frame
        person = cv2.bitwise_and(frame, frame, mask=mask)

        # Send the frame to the virtual camera
        cam.send(person)
        cam.sleep_until_next_frame()
