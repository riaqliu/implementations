from __future__ import annotations
from collections import namedtuple
import random

TrieNode = namedtuple('TrieNode', ['value','score','children'])

def create_trie_node(value, score = None):
    return TrieNode(value, score, [])

def insert_trie_node_child(parent:TrieNode, child:TrieNode):
    parent.children.append(child)

def insert_nodes(string, head:TrieNode = None, score = -1):
    if len(string) > 0:
        flag = False
        for child in head.children:
            child:TrieNode
            if child.value == string[0]:
                flag = True
                insert_nodes(string[1:], child)
                break
        if not flag:
            child = create_trie_node(string[0], score)
            insert_trie_node_child(head, child)
            insert_nodes(string[1:], child)

def get_value(string, head:TrieNode):
    if len(string) > 0:
        for child in head.children:
            child:TrieNode
            if string[0] == child.value:
                if len(string) == 1:
                    return child.score
                return get_value(string[1:], child)
    return None

def print_nodes(head:TrieNode, string = "", level = 0):
    string += head.value
    print(level, string)
    for child in head.children:
        child:TrieNode
        print_nodes(child, string, level+1)

if __name__ == "__main__":
    # head = create_trie_node('', None)
    # insert_nodes('HELLO', head)

    # print_nodes(head)
    # print(head)
    # print(get_value('HELLI', head))

    pass
    import sys

    my_dict = {'key1': 123, 'key2': 'value2', 'key3': [1, 2, 3], 'aa':2}

    size_in_bytes = sys.getsizeof(my_dict)
    print("Size of dictionary:", size_in_bytes, "bytes")

