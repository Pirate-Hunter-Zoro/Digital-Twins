from enum import Enum
import os

from dotenv import load_dotenv
load_dotenv()

class NeighborScheme(Enum):
    NEAREST = "nearest"
    FARTHEST = "farthest"
    RANDOM = "random"
    SUBSAMPLE = "subsample"

RELEVANT_NEIGHBOR_SCHEMES = [scheme for scheme in NeighborScheme]
if int(os.environ['NEIGHBOR_FARTHEST']) == 0:
    RELEVANT_NEIGHBOR_SCHEMES.remove(NeighborScheme.FARTHEST)
if int(os.environ['NEIGHBOR_SUBSAMPLE']) == 0:
    RELEVANT_NEIGHBOR_SCHEMES.remove(NeighborScheme.SUBSAMPLE)