from enum import Enum

class NeighborScheme(Enum):
    NEAREST = "nearest"
    FARTHEST = "farthest"
    RANDOM = "random"
    SUBSAMPLE = "subsample"