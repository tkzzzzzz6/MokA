
import accelerate
import sys
print("Python executable:", sys.executable)
print("Python version:", sys.version)
print("accelerate version:", accelerate.__version__)
print("accelerate path:", accelerate.__file__)

# Check if dispatch_batches is a parameter in Accelerator.__init__
import inspect
from accelerate import Accelerator
sig = inspect.signature(Accelerator.__init__)
print("\nAccelerator.__init__ parameters:", list(sig.parameters.keys()))
print("'dispatch_batches' in parameters:", 'dispatch_batches' in sig.parameters)
