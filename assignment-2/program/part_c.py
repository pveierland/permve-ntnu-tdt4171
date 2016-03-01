#!/usr/bin/python3

import vi.hmm
import numpy

model = vi.hmm.hmm(
    transition_model=numpy.array([[0.7, 0.3], [0.3, 0.7]]),
    sensor_model={
        True: numpy.array([[0.9, 0.0], [0.0, 0.2]]),
        False: numpy.array([[0.1, 0.0], [0.0, 0.8]]) },
    message=numpy.array([0.5, 0.5]))

evidence_values = [True, True, False, True, True]
result = model.forward_backward(evidence_values, model.message)

for t, (forward_value, backward_value, smooth_value) in enumerate(zip(*result)):
    print(('{} & \\textit{{{}}}' + 3 * '& $\\langle {:.3f}, {:.3f} \\rangle$' + ' \\\\').format(
        t,
        str(evidence_values[t - 1]).lower() if t > 0 else '-',
        forward_value[0],
        forward_value[1],
        backward_value[0],
        backward_value[1],
        smooth_value[0],
        smooth_value[1]))

