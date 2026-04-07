import torch
print('CUDA available:', torch.cuda.is_available())
print('CUDA device count:', torch.cuda.device_count())
if torch.cuda.is_available():
    print('Current device:', torch.cuda.current_device())
    print('Device name:', torch.cuda.get_device_name())
