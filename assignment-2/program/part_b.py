#!/usr/bin/python3

import vi.hmm
import numpy

model = vi.hmm.hmm(
    transition_model=numpy.array([[0.7, 0.3], [0.3, 0.7]]),
    sensor_model={
        True: numpy.array([[0.9, 0.0], [0.0, 0.2]]),
        False: numpy.array([[0.1, 0.0], [0.0, 0.8]]) },
    message=numpy.array([0.5, 0.5]))

forward_messages = [model.message]
evidence_values  = [True, True, False, True, True]

for evidence_value in evidence_values:
    forward_messages.append(model.forward(forward_messages[-1], evidence_value))

for i, forward_message in enumerate(forward_messages):
    print('{0} & \\textit{{{3}}} & $\\langle {1:.3f}, {2:.3f} \\rangle$ \\\\'.format(
        i,
        forward_message[0],
        forward_message[1],
        str(evidence_values[i - 1]).lower() if i > 0 else '-'))

