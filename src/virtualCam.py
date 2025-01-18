import cv2
import numpy as np
import pyvirtualcam
from pyvirtualcam import PixelFormat
from tflite_runtime.interpreter import Interpreter, load_delegate
from pycoral.adapters import common, segment

# Paths to the model and resources
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

# Load the background image
background = cv2.imread(BACKGROUND_IMAGE_PATH)
if background is None:
    print("Error: Could not load background image.")
    exit()

# Open the webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Get webcam dimensions
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Resize the background image to match the webcam dimensions
background = cv2.resize(background, (frame_width, frame_height))

# Initialize the virtual camera
with pyvirtualcam.Camera(width=frame_width, height=frame_height, fps=30, fmt=PixelFormat.BGR) as cam:
    print(f'Using virtual camera: {cam.device}')
    print("Press 'Ctrl+C' to exit.")

    while True:
        # Capture a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture video frame.")
            break

        # Resize the frame to the model input size for inference
        resized_frame = cv2.resize(frame, input_size)

        # Prepare the input for the model
        common.set_input(interpreter, resized_frame)

        # Run inference
        interpreter.invoke()

        # Get the segmentation result
        result = segment.get_output(interpreter)
        if len(result.shape) == 3:
            result = np.argmax(result, axis=-1)

        # Print unique values in the segmentation result
        # print(f"Segmentation result unique values: {np.unique(result)}")

        # Create a binary mask for the 'person' class (index 15)
        mask = (result == 15).astype(np.uint8) * 255

        # Display the mask
        #cv2.imshow("Mask", mask)

        # Resize the mask to match the webcam dimensions
        mask = cv2.resize(mask, (frame_width, frame_height), interpolation=cv2.INTER_NEAREST)
        mask = cv2.GaussianBlur(mask, (7, 7), 0)

        # Ensure the mask is three channels to match the frame and background
        mask_3channel = cv2.merge([mask, mask, mask])

        # Create an inverse mask
        inverse_mask = cv2.bitwise_not(mask)

        # Display the inverse mask, testing
        # cv2.imshow("Inverse Mask", inverse_mask)

        # Ensure the inverse mask is three channels
        inverse_mask_3channel = cv2.merge([inverse_mask, inverse_mask, inverse_mask])

        # Extract the person from the frame using the mask
        person_segment = cv2.bitwise_and(frame, mask_3channel)

        # Extract the background using the inverse mask
        background_segment = cv2.bitwise_and(background, inverse_mask_3channel)

        # Combine the person and the background
        combined_frame = cv2.add(person_segment, background_segment)

        # Display the combined frame (nice for testing)
        # cv2.imshow("Combined Frame", combined_frame)

        # Send the combined frame to the virtual camera
        cam.send(combined_frame)
        cam.sleep_until_next_frame()

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the webcam
cap.release()
cv2.destroyAllWindows()