import cv2
import numpy as np
import pyvirtualcam
from pyvirtualcam import PixelFormat
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters import common, segment

MODEL_PATH = '../models/deeplabv3_mnv2_pascal_quant_edgetpu.tflite'
BACKGROUND_IMAGE_PATH = '../models/background.jpg'

# Initialize the interpreter
interpreter = make_interpreter(MODEL_PATH)
interpreter.allocate_tensors()
input_size = common.input_size(interpreter)

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
        mask = cv2.GaussianBlur(mask, (7, 7), 0)

        # Resize the background to match the original frame size
        background_resized = cv2.resize(background, (frame.shape[1], frame.shape[0]))

        # Create an inverse mask
        inverse_mask = cv2.bitwise_not(mask)

        # Apply the mask to the frame
        person = cv2.bitwise_and(frame, frame, mask=mask)

        # Extract the background where the person is not present
        background_segment = cv2.bitwise_and(background_resized, background_resized, mask=inverse_mask)

        # Combine the person and the new background
        combined = cv2.add(person, background_segment)

        # Send the frame to the virtual camera
        cam.send(combined)
        cam.sleep_until_next_frame()
