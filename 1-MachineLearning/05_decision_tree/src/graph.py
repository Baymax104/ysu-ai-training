# -*- coding: UTF-8 -*-

from graphviz import Digraph


def level_trace(tree, feature_names, class_names):
    graph = Digraph(format='png', comment='Decision Tree', name='Decision Tree')
    graph.graph_attr['label'] = 'Decision Tree'
    graph.graph_attr['dpi'] = '200'

    queue = [(0, tree.root)]
    while len(queue) > 0:
        node_id, node = queue.pop(0)

        # leaf node
        if node.value is not None:
            graph.node(f'{node_id}', label=f'Class: {class_names[node.value]}', shape='box')
        else:
            graph.node(f'{node_id}', label=f'{feature_names[node.feature]} <= {node.threshold:.4f}', shape='box')

            left_id = node_id * 2 + 1
            right_id = node_id * 2 + 2

            if node.left is not None:
                graph.edge(f'{node_id}', f'{left_id}', label=f'True')
                queue.append((left_id, node.left))

            if node.right is not None:
                graph.edge(f'{node_id}', f'{right_id}', label=f'False')
                queue.append((right_id, node.right))

    return graph


def export_tree(tree, feature_names, class_names):
    graph = level_trace(tree, feature_names, class_names)
    graph.render('tree.gv', directory='../logs', format='png')
