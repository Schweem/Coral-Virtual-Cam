import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter, load_delegate
from pycoral.adapters import common, segment

# Paths to the model and background image
MODEL_PATH = '../models/deeplabv3_mnv2_pascal_quant_edgetpu.tflite'
BACKGROUND_IMAGE_PATH = '../models/background.jpg'
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

# Load and resize the background image to match the input size
background = cv2.imread(BACKGROUND_IMAGE_PATH)
if background is None:
    print("Error: Could not load background image.")
    exit()
background = cv2.resize(background, input_size)

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

    # Create a binary mask where the 'person' class (index 15) is detected
    mask = (result == 15).astype(np.uint8) * 255

    # Resize the mask and background to match the original frame size
    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
    mask = cv2.GaussianBlur(mask, (7, 7), 0)

    background_resized = cv2.resize(background, (frame.shape[1], frame.shape[0]))

    # Create an inverse mask
    inverse_mask = cv2.bitwise_not(mask)

    # Extract the person from the frame using the mask
    person = cv2.bitwise_and(frame, frame, mask=mask)

    # Extract the background where the person is not present
    background_segment = cv2.bitwise_and(background_resized, background_resized, mask=inverse_mask)

    # Combine the person and the new background
    combined = cv2.add(person, background_segment)

    # Display the result
    cv2.imshow('Green Screen Effect', combined)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()