from tinynet import RNN1L
import numpy as np
ninputs, noutputs = [3, 2]
net = RNN1L(ninputs, noutputs)
net.set_weights(np.random.rand(net.nweights()))
out = net.activate(np.zeros(ninputs))
assert len(out) == noutputs
assert len(net.state) == ninputs + 1 + noutputs # input, bias, recursion
