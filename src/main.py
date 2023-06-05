import tensorflow as tf
import tensorflow as tf

gpu_devices = tf.config.list_physical_devices('GPU')
if len(gpu_devices) > 0:
  print(f'{len(gpu_devices)} GPU(s) found and ready for use👍')
  print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
else:
  print('No GPU support☹')