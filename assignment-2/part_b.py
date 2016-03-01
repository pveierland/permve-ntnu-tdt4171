#!/usr/bin/python3

import numpy

def forward(message, evidence):
    return sensor_model[evidence].dot(numpy.transpose(transition_model)).dot(message)

def normalize(x):
    n = sum(x)
    return x / n if n else x

transition_model = numpy.array([[0.7, 0.3], [0.3, 0.7]])

sensor_model = {
    True: numpy.array([[0.9, 0.0], [0.0, 0.2]]),
    False: numpy.array([[0.1, 0.0], [0.0, 0.8]])
}

forward_messages = [numpy.array([0.5, 0.5])]
evidence = [True, True, False, True, True]

for e in evidence:
    forward_messages.append(normalize(forward(forward_messages[-1], e)))

for i, forward_message in enumerate(forward_messages):
    print('{0} & \\textit{{{3}}} & $\\langle {1:.3f}, {2:.3f} \\rangle$ \\\\'.format(
        i,
        forward_message[0],
        forward_message[1],
        str(evidence[i - 1]).lower() if i > 0 else '-'))

