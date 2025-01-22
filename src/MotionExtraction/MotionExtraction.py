import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter, load_delegate
import os
from collections import deque

# Initialize Edge TPU
MODEL_PATH = os.path.join('models', 'ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite')
EDGETPU_SHARED_LIB = '../tpulib/libedgetpu.1.dylib'

# Load the Edge TPU model for frame processing
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

def process_frame_tpu(frame):
    """Process frame through TPU for potential enhancement"""
    input_shape = input_details[0]['shape']
    processed = cv2.resize(frame, (input_shape[2], input_shape[1]))
    
    input_data = np.expand_dims(processed, axis=0)
    if input_details[0]['dtype'] == np.uint8:
        input_data = input_data.astype(np.uint8)
    
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    output_data = interpreter.get_tensor(output_details[0]['index'])
    processed_frame = cv2.resize(processed, (frame.shape[1], frame.shape[0]))
    return processed_frame

def apply_rgb_motion_extraction(current_frame, delayed_frame, channel_offsets, blend_alpha):
    """Apply motion extraction with different offsets for RGB channels"""
    result = np.zeros_like(current_frame)
    
    for channel in range(3):
        current_channel = current_frame[:, :, channel]
        delayed_channel = delayed_frame[:, :, channel]
        inverted_channel = cv2.bitwise_not(delayed_channel)
        
        if channel_offsets[channel] != 0:
            rows, cols = inverted_channel.shape
            M = np.float32([[1, 0, channel_offsets[channel]], [0, 1, 0]])
            inverted_channel = cv2.warpAffine(inverted_channel, M, (cols, rows))
        
        result[:, :, channel] = cv2.addWeighted(
            current_channel,
            1.0,
            inverted_channel,
            blend_alpha,
            0
        )
    
    return result

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Parameters
frame_buffer_size = 15
frame_buffer = deque(maxlen=frame_buffer_size)
blend_alpha = 0.5
channel_offsets = [0, 0, 0]  # RGB channel offsets
effect_mode = 0  # 0: normal, 1: RGB offset

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
        
    processed_frame = process_frame_tpu(frame)
    frame_buffer.append(processed_frame)
    
    if len(frame_buffer) >= frame_buffer_size:
        delayed_frame = frame_buffer[0]
        
        if effect_mode == 0:
            # Normal motion extraction
            inverted_frame = cv2.bitwise_not(delayed_frame)
            motion_highlight = cv2.addWeighted(
                processed_frame,
                1.0,
                inverted_frame,
                blend_alpha,
                0
            )
        else:
            # RGB channel motion extraction
            motion_highlight = apply_rgb_motion_extraction(
                processed_frame,
                delayed_frame,
                channel_offsets,
                blend_alpha
            )
        
        # Add visual information
        info_text = [
            f"Mode: {'RGB' if effect_mode else 'Normal'}",
            f"Frame Offset: {frame_buffer_size}",
            f"Blend: {blend_alpha:.2f}",
            f"R:{channel_offsets[0]} G:{channel_offsets[1]} B:{channel_offsets[2]}"
        ]
        
        for i, text in enumerate(info_text):
            cv2.putText(motion_highlight, text, 
                       (10, 30 + (i * 30)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.8, 
                       (0, 255, 0), 
                       2)
        
        cv2.imshow('RGB Motion Extraction', motion_highlight)
    
    # Keyboard controls with simpler key mapping
    key = cv2.waitKey(1) & 0xFF
    
    # Basic controls
    if key == ord('q'):
        break
    elif key == ord('m'):
        effect_mode = 1 - effect_mode
    elif key == ord(' '):  # space
        channel_offsets = [0, 0, 0]
        blend_alpha = 0.5
        frame_buffer_size = 15
        frame_buffer = deque(maxlen=frame_buffer_size)
    
    # Frame offset controls
    elif key == ord('a'):  # decrease offset
        if frame_buffer_size > 2:
            frame_buffer_size -= 1
            frame_buffer = deque(maxlen=frame_buffer_size)
    elif key == ord('s'):  # increase offset
        if frame_buffer_size < 30:
            frame_buffer_size += 1
            frame_buffer = deque(maxlen=frame_buffer_size)
    
    # Blend controls
    elif key == ord('z'):  # decrease blend
        blend_alpha = max(0.1, blend_alpha - 0.1)
    elif key == ord('x'):  # increase blend
        blend_alpha = min(1.0, blend_alpha + 0.1)
    
    # RGB channel controls
    elif key == ord('r'):  # decrease red offset
        channel_offsets[0] = max(-50, channel_offsets[0] - 2)
    elif key == ord('t'):  # increase red offset
        channel_offsets[0] = min(50, channel_offsets[0] + 2)
    elif key == ord('g'):  # decrease green offset
        channel_offsets[1] = max(-50, channel_offsets[1] - 2)
    elif key == ord('h'):  # increase green offset
        channel_offsets[1] = min(50, channel_offsets[1] + 2)
    elif key == ord('b'):  # decrease blue offset
        channel_offsets[2] = max(-50, channel_offsets[2] - 2)
    elif key == ord('n'):  # increase blue offset
        channel_offsets[2] = min(50, channel_offsets[2] + 2)

# Cleanup
cap.release()
cv2.destroyAllWindows()