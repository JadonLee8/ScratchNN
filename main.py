import Sigmoids
import numpy as np
from nnetwork import network

if __name__ == "__main__":
    net = network([2, 2, 1], 2)
    net.feed_forward([1, 1])