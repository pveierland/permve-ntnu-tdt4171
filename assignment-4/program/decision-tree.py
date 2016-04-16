#!/usr/bin/python3

import argparse
import math
import operator
import random

from collections import Counter

def entropy(examples, target_attribute):
    outcome_counts = Counter(example.assignments[target_attribute] for example in examples)
    total_outcomes = len(examples)

    entropy_sum = 0

    for value, count in outcome_counts.items():
        outcome_probability = count / total_outcomes
        entropy_sum -= outcome_probability * math.log2(outcome_probability)

    return entropy_sum

def entropic_attribute_evaluator(examples, attribute, target_attribute):
    remaining_entropy = 0

    for value in attribute.values:
        e = list(filter(lambda example: example.assignments[attribute] == value, examples))
        remaining_entropy += len(e) / len(examples) * entropy(e, target_attribute)

    current_entropy  = entropy(examples, target_attribute)
    information_gain = current_entropy - remaining_entropy

    #print('attribute {} has information gain {}'.format(attribute.name, information_gain))

    return information_gain

def plurality_value(examples, target_attribute):
    values         = [example.assignments[target_attribute] for example in examples]
    values_counted = sorted(((values.count(value), value) for value in set(values)), reverse=True, key=operator.itemgetter(0))
    most_common    = list(filter(lambda value_counted: value_counted[0] == values_counted[0][0], values_counted))

    # print('selecting plurality value from: {}'.format(most_common))

    return random.choice(most_common)[1]

def random_attribute_evaluator(examples, attribute, target_attribute):
    return random.random()

def read_file(input_file, attributes=None, examples=None):
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

    attributes = attributes if attributes is not None else []
    examples   = examples if examples is not None else []

    with open(input_file) as data:
        first_line = data.readline()

        if first_line.startswith('#') and not attributes:
            attributes = read_attributes(first_line)
        else:
            examples.append(read_example(first_line))

        for line in data:
            examples.append(read_example(line))

    return attributes, examples

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
              parent_examples     = None,
              attribute_evaluator = entropic_attribute_evaluator,
              ambiguity_resolver  = plurality_value):

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
            attributes_scored = sorted(((attribute_evaluator(examples, attribute, target_attribute), attribute)
                                        for attribute in attributes), reverse=True, key=operator.itemgetter(0))
            attributes_scored = list(filter(lambda a: a[0] == attributes_scored[0][0], attributes_scored))

            # if len(attributes_scored) > 1:
            #    print('attributes scored equal: {}'.format(','.join(a[1].name for a in attributes_scored)))

            attribute = random.choice(attributes_scored)[1]

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

    def render_dot(self, dot=None):
        is_root = not dot

        if is_root:
            dot = { 'node_id': 0 }

        node_name       = 'N{}'.format(dot['node_id'])
        node_dot        = '{} [label="{}"];\n'.format(node_name, self.attribute.name)
        dot['node_id'] += 1

        for value, child in self.branches.items():
            child_node_name, child_dot = child.render_dot(dot)
            node_dot += child_dot + '{} -> {} [label="{}"];\n'.format(node_name, child_node_name, value)

        if is_root:
            return node_dot
        else:
            return node_name, node_dot

class DecisionValueNode(object):
    def __init__(self, attribute, value):
        self.attribute = attribute
        self.value     = value

    def __call__(self, example):
        return self.value

    def render_dot(self, dot):
        node_name       = 'N{}'.format(dot['node_id'])
        node_dot        = '{} [label="{}"];\n'.format(node_name, self.value)
        dot['node_id'] += 1
        return (node_name, node_dot)

parser = argparse.ArgumentParser()
parser.add_argument("training_file")
parser.add_argument('--all', action='store_true')
parser.add_argument('--dot', action='store_true')
parser.add_argument('--random', action='store_true')
parser.add_argument('--target')
parser.add_argument('--test_file')
arguments = parser.parse_args()

attributes, training_examples = read_file(arguments.training_file)

if arguments.target:
    target_attribute = next(
        training_attribute
        for training_attribute in attributes
        if training_attribute.name == arguments.target)
else:
    target_attribute = attributes[-1]

test_examples = []

if arguments.test_file:
    read_file(arguments.test_file, attributes, test_examples)

if arguments.all:
    training_examples = training_examples + test_examples
    test_examples     = test_examples + training_examples

attribute_evaluator = random_attribute_evaluator if arguments.random else entropic_attribute_evaluator

decision_tree = DecisionAttributeNode.build(
    training_examples,
    set(attributes) - set([target_attribute]),
    target_attribute,
    attribute_evaluator=attribute_evaluator)

if arguments.dot:
    print(decision_tree.render_dot())

if arguments.test_file or arguments.all:
    correct = sum(
        test_example.assignments[target_attribute] == decision_tree(test_example)
        for test_example in test_examples)

    print('{}/{} ({}%) correctly classified'.format(
        correct, len(test_examples), round(100 * correct / len(test_examples), 2)))

