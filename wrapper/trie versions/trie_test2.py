import string
from numpy import mean
from sklearn.model_selection import cross_val_score

class TrieNode():
    def __init__(self) -> None:
        self.end = False
        self.children = [None for _ in range(26)]

def node_insert_key(root:TrieNode, key):
    ascii = string.ascii_lowercase
    currentNode:TrieNode = root
    for c in key:
        n = ascii.find(c)
        if currentNode.children[n] == None:
            newNode = TrieNode()
            currentNode.children[n] = newNode
        currentNode = currentNode.children[n]
    currentNode.end = True

def node_search_key(root:TrieNode, key):
    ascii = string.ascii_lowercase
    currentNode:TrieNode = root
    for c in key:
        n = ascii.find(c)
        if currentNode.children[n] == None:
            return False
        currentNode = currentNode.children[n]

    return currentNode.end == True

if __name__ == '__main__':
    top = TrieNode()
    node_insert_key(top, "hello triplet")
    print(node_search_key(top, "hello triplet"))