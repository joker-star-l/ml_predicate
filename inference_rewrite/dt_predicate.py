import onnx
import time
import pandas as pd
from typing import List, Tuple, Dict
import argparse
from tree import model2tree
import sys
            
class Predicate:
    def __init__(
        self,
        feature_id,
        lvalue,
        rvalue
        ):
        self.feature_id: int = feature_id
        self.lvalue: float | None = lvalue
        self.rvalue: float | None = rvalue

def get_attribute(onnx_model, attr_name):
    attributes = onnx_model.graph.node[0].attribute
    for attr in attributes:
        if attr.name == attr_name:
            return attr
        
def get_feature_ids(input_model):
    input_nodes_featureids = get_attribute(input_model, 'nodes_featureids').ints
    return list(set(input_nodes_featureids))


def simplify_predicates_disjunction(predicates) -> 'Predicate | None':
    min_value = float('inf')
    max_value = float('-inf')

    for p in predicates:
        if p == None:
            return None
        print(f'disjunction_before_feature_id: {p.feature_id}, lvalue: {p.lvalue}, rvalue: {p.rvalue}')
        if p.lvalue is not None and p.lvalue > max_value:
            max_value = p.lvalue
        if p.rvalue is not None and p.rvalue < min_value:
            min_value = p.rvalue
        print(f'disjunction_before_feature_id: {p.feature_id}, max_value: {max_value}, min_value: {min_value}')
    print(f'disjunction_after_feature_id: {predicates[0].feature_id}, lvalue: {max_value}, rvalue: {min_value}\n')
    return Predicate(predicates[0].feature_id, max_value, min_value)

def simplify_predicates_conjunction(predicates) -> 'Predicate | None':
    min_value = float('inf')
    max_value = float('-inf')

    for p in predicates:
        if p == None:
            return None
        print(f'conjunction_before_feature_id: {p.feature_id}, lvalue: {p.lvalue}, rvalue: {p.rvalue}')
        if p.lvalue is not None and p.lvalue < min_value:
            min_value = p.lvalue
        if p.rvalue is not None and p.rvalue > max_value:
            max_value = p.rvalue
    print(f'conjunction_after_feature_id: {predicates[0].feature_id}, lvalue: {min_value}, rvalue: {max_value}\n')
    return Predicate(predicates[0].feature_id, min_value, max_value)

def generate_predicate(root:'Node', feature_id, f) -> 'Predicate | None':
    Predicates = []
    stack = [root]
    while stack:
        node = stack.pop()
        if node.right:
            stack.append(node.right)
        if node.left:
            stack.append(node.left)                
        if node.mode == b'LEAF':
            if (int(f(node.target_weight))):
                predicates = []
                curr = node
                while curr.parent is not None:
                    parent = curr.parent
                    child = curr
                    if parent.feature_id == feature_id:
                        if child.id == parent.left.id:
                            predicates.append(Predicate(feature_id, parent.value, float('-inf')))
                        else:
                            predicates.append(Predicate(feature_id, float('inf'), parent.value))
                    curr = parent
                if predicates:
                    Predicates.append(simplify_predicates_conjunction(predicates))
    if Predicates:
        return simplify_predicates_disjunction(Predicates)
    else:
        return None

def generate_predicates(input_model, root:'Node', f) -> 'List[Predicate | None]':
    predicates = []
    feature_ids = get_feature_ids(input_model)
    print(feature_ids)
    for feature_id in feature_ids:
        predicates.append(generate_predicate(root, feature_id, f))
        print('\n')
        # break
    return predicates    
    
parser = argparse.ArgumentParser()
parser.add_argument('--model', '-m', type=str, default='nyc-taxi-green-dec-2016_d10_l449_n897_20240903082951')
parser.add_argument('--threshold', '-t', type=int, default=3.1)
parser.add_argument('--data', '-d', type=str, default='nyc-taxi-green-dec-2016')
args = parser.parse_args()

model_name = args.model
threshold = args.threshold
data_name = args.data

func = lambda x: x > threshold

model_path = f'../model/{model_name}.onnx'
model = onnx.load(model_path)

root = model2tree(model, None, 0, None, None, None)
root.parent = None

predicates = generate_predicates(model, root, func)

data_path = f'/volumn/ml_predicate/data/{data_name}.csv'
data = pd.read_csv(data_path)

for p in predicates:
    count = 0
    correct = 0
    incorrect = 0
    if p is not None:
        for _, row in data.iterrows():
            feature_value = row.iloc[p.feature_id]
            if feature_value < p.lvalue and feature_value > p.rvalue:
                count += 1
                if row.iloc[-1] > threshold:
                    correct += 1
            else:
                if row.iloc[-1] > threshold:
                    incorrect += 1
        print(f'feature_id: {p.feature_id}, lvalue: {p.lvalue}, rvalue: {p.rvalue}, count: {count}, satisfy_scale: {round(count/data.shape[0],2)}, correct: {correct}, incorrect: {incorrect}')
                 
    



    