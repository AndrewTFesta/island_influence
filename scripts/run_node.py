import time

from island_influence.learn.island import node

node0_port = 65434
node1_port = 65435

node0 = node.Node('node0', '', node0_port)
node1 = node.Node('node1', '', node1_port)

node0.start()
node1.start()

node0.connect_to('127.0.0.1', node1_port)
node1.connect_to('127.0.0.1', node0_port)


def test_node_connect():
    print(f'Number of node0 connections: {len(node0.nodes_connected)}')
    print(f'Number of node1 connections: {len(node1.nodes_connected)}')
    return


def test_node_message():
    node0.send_message('node test')
    print(f'Number of node0 messages: {len(node0.msgs.keys())}')
    print(f'Number of node1 messages: {len(node1.msgs.keys())}')
    return


test_node_connect()
test_node_message()

time.sleep(2)

node0.stop()
node1.stop()
