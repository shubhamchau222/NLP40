## All the constants
import torch
DEVICE= "cuda" if torch.cuda.is_available() else "cpu"

VOCAB_SIZE= 20257
CONTEXT_LENGTH = 256
D_MODEL= 512 # Embeddibg Dimentins ( Hello: [..............]: represented by 512dims)
N_HEADS= 8
N_LAYERS = 6
DROPOUT= 0.1
D_FF = D_MODEL*4 # Intermidiate dims
EPSILON = 1e-5 # epsilon for layer norm