import Sigmoids
import numpy as np
from nnetwork import network

if __name__ == "__main__":
    net = network([16, 16, 4], 748)
    net.feed_forward(np.random.random(size=748))