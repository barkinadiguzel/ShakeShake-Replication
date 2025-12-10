# Model parameters
NUM_CLASSES = 10
IN_CHANNELS = 3
BASE_CHANNELS = 16
NUM_BLOCKS = [4, 4, 4]  

# Training parameters
BATCH_SIZE = 128
LEARNING_RATE = 0.2
EPOCHS = 1800
DEVICE = 'cuda' 

# Shake-Shake parameters
SHAKE_LEVEL = 'image'  # 'batch' or 'image'
