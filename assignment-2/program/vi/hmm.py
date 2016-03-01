import numpy

class hmm(object):
    @staticmethod
    def normalize(x):
        return x / x.sum()

    def __init__(self, transition_model, sensor_model, message):
        self.transition_model = transition_model
        self.sensor_model     = sensor_model
        self.message          = message

    def backward(self, message, evidence):
        return self.transition_model.dot(
            self.sensor_model[evidence].dot(
                message))

    def forward(self, message, evidence):
        return self.normalize(
            self.sensor_model[evidence].dot(
                numpy.transpose(self.transition_model).dot(
                    message)))

    def forward_backward(self, evidence_values, prior):
        t = len(evidence_values)

        forward_values = [prior]
        for i in range(1, t + 1):
            forward_values.append(self.forward(forward_values[i - 1], evidence_values[i - 1]))

        backward_values = [numpy.ones(prior.shape)]
        smooth_values   = [forward_values[-1]]

        for forward_value, evidence_value in zip(reversed(forward_values[:-1]), reversed(evidence_values)):
            backward_value = self.backward(backward_values[0], evidence_value)
            backward_values.insert(0, backward_value)
            smooth_values.insert(0, self.normalize(forward_value * backward_value))

        return forward_values, backward_values, smooth_values

    def step_backward(self, evidence):
        self.message = self.backward(self.message, evidence)
        return self.message

    def step_forward(self, evidence):
        self.message = self.forward(self.message, evidence)
        return self.message

