import platform
from pycoral.utils.edgetpu import list_edge_tpus
from pycoral.adapters.common import input_size
from tflite_runtime.interpreter import Interpreter, load_delegate

def test_coral():
    
    # List available Edge TPUs
    devices = list_edge_tpus()
    
    if not devices:
        print("No Edge TPU devices detected. Make sure the Coral device is connected.")
        return
    print("Edge TPU devices detected:")
    
    for device in devices:
        print(f"  {device}")

    # Having issues with libary path, directly linking for now (temporary)
    # Change to yours or remove if i am just not smart enough to figure it out
    
    edgetpu_library_path = "../tpulib/libedgetpu.1.dylib"

    # Load the model
    model_path = "../../models/deeplabv3_mnv2_pascal_quant_edgetpu.tflite"
    
    try:
        # Create interpreter with Edge TPU delegate
        interpreter = Interpreter(
            model_path=model_path, 
            experimental_delegates=[load_delegate(edgetpu_library_path)]
        )
        
        interpreter.allocate_tensors()
        print(f"Model '{model_path}' loaded successfully.")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        
        return

    # Display input size of the model
    try:
        input_details = interpreter.get_input_details()
        
        if input_details:
            input_shape = input_details[0]['shape']
            print(f"Model input size: {input_shape}")
        else:
            print("No input details found in the model.")
            
    except Exception as e:
        print(f"Error fetching input size: {e}")

if __name__ == "__main__":
    test_coral()
