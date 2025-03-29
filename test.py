import torch

print("CUDA disponível:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("Nome da GPU:", torch.cuda.get_device_name(0))
    print("Número de GPUs:", torch.cuda.device_count())
    print("Versão do driver CUDA:", torch.version.cuda)
    print("Versão do PyTorch:", torch.__version__)
else:
    print("CUDA não está a funcionar corretamente com PyTorch.")

import tensorflow as tf

print("TensorFlow version:", tf.__version__)
print("GPU disponível:", tf.config.list_physical_devices('GPU'))

