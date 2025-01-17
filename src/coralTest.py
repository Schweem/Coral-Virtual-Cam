from pycoral.utils.edgetpu import list_edge_tpus
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters.common import input_size

def test_coral():
    # List available Edge TPUs
    devices = list_edge_tpus()
    if not devices:
        print("No Edge TPU devices detected.")
        return
    print("Edge TPU devices detected:")
    for device in devices:
        print(f"  {device}")

    # Load the model
    model_path = "../models/deeplabv3_mnv2_pascal_quant_edgetpu.tflite"
    try:
        interpreter = make_interpreter(model_path)
        interpreter.allocate_tensors()
        print(f"Model '{model_path}' loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Display input size of the model
    input_shape = input_size(interpreter)
    print(f"Model input size: {input_shape}")

if __name__ == "__main__":
    test_coral()
