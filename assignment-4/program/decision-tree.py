#!/usr/bin/python3

import argparse
import math
import operator
import random

from collections import Counter

def plurality_value(examples, target_attribute):
    values         = [example.assignments[target_attribute] for example in examples]
    values_counted = sorted(((values.count(value), value) for value in set(values)), reverse=True)
    most_common    = list(filter(lambda value_counted: value_counted[0] == values_counted[0][0], values_counted))
    return random.choice(most_common)

def entropy(examples, target_attribute):
    outcome_counts = Counter(example.assignments[target_attribute] for example in examples)
    total_outcomes = len(examples)

    entropy_sum = 0

    for value, count in outcome_counts.items():
        outcome_probability = count / total_outcomes
        entropy_sum -= outcome_probability * math.log2(outcome_probability)

    return entropy_sum

def entropic_information_gain(examples, attribute, target_attribute):
    remaining_entropy = 0

    for value in attribute.values:
        e = list(filter(lambda example: example.assignments[attribute] == value, examples))
        remaining_entropy += len(e) / len(examples) * entropy(e, target_attribute)

    current_entropy  = entropy(examples, target_attribute)
    information_gain = current_entropy - remaining_entropy

    return information_gain

class Attribute(object):
    def __init__(self, name, values=None):
        self.name   = name
        self.values = values or set()

class Example(object):
    def __init__(self, name, assignments=None):
        self.name        = name
        self.assignments = assignments or {}

    def __str__(self):
        return 'Example(name={},assignments={})'.format(
            self.name, (','.join('{}:{}'.format(attribute.name, value)
            for (attribute, value) in self.assignments.items())))

class DecisionAttributeNode(object):
    @staticmethod
    def build(examples,
              attributes,
              target_attribute,
              parent_examples      = None,
              attribute_evalutator = entropic_information_gain,
              ambiguity_resolver   = plurality_value):

        if not examples:
            return DecisionValueNode(
                target_attribute, ambiguity_resolver(parent_examples, target_attribute))
        elif all(example.assignments[target_attribute] == examples[0].assignments[target_attribute]
                 for example in examples[1:]):
            return DecisionValueNode(
                target_attribute, examples[0].assignments[target_attribute])
        elif not attributes:
            return DecisionValueNode(
                target_attribute, ambiguity_resolver(examples, target_attribute))
        else:
            attribute = max(attributes,
                key=lambda attribute: attribute_evalutator(examples, attribute, target_attribute))

            attribute_node = DecisionAttributeNode(attribute)

            for value in attribute.values:
                e = [example for example in examples if example.assignments[attribute] == value]

                attribute_node.branches[value] = DecisionAttributeNode.build(
                    e, attributes - set([attribute]), target_attribute, examples)

            return attribute_node

    def __init__(self, attribute, branches=None):
        self.attribute = attribute
        self.branches  = branches or {}

    def __call__(self, example):
        return self.branches[example.assignments[self.attribute]](example)

class DecisionValueNode(object):
    def __init__(self, attribute, value):
        self.attribute = attribute
        self.value     = value

    def __call__(self, example):
        return self.value

def read_file(input_file):
    def get_attribute(index):
        while index >= len(attributes):
            attributes.append(Attribute('Attribute #{}'.format(len(attributes) + 1)))
        return attributes[index]

    def read_attributes(line):
        return [Attribute(name) for name in line[1:].split()]

    def read_example(line):
        example = Example('Example #{}'.format(len(examples) + 1))

        for attribute_index, value in enumerate(line.split()):
            attribute = get_attribute(attribute_index)
            attribute.values.add(value)
            example.assignments[attribute] = value

        return example

    attributes = []
    examples   = []

    with open(input_file) as data:
        first_line = data.readline()

        if first_line.startswith('#'):
            attributes = read_attributes(first_line)
        else:
            examples.append(read_example(first_line))

        for line in data:
            examples.append(read_example(line))

    return attributes, examples


parser = argparse.ArgumentParser()
parser.add_argument("training_file")
#parser.add_argument('--test', 'Test data file path')
arguments = parser.parse_args()

training_attributes, training_examples = read_file(arguments.training_file)
decision_tree = DecisionAttributeNode.build(
    training_examples, set(training_attributes[:-1]), training_attributes[-1])



print(decision_tree(Example('e', { training_attributes[4]: 'Some' })))


#if arguments['test_file']:
#    test_attributes, test_examples = read_file(arguments['test_file'])
#
#    sum(decision_tree(test_example) == test_example.assignments[
#
#    for test_example in test_examples:
#        classification = decision_tree(test_example)


#for attribute in attributes[:-1]:
#    print('attribute = {} information gain = {}'.format(attribute.name, entropic_information_gain(examples, attribute, attributes[-1])))

