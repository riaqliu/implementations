class SBTN():
    '''
        Scored Bit Trie Node
    '''
    features = '01'
    def __init__(self) -> None:
        self.end = False
        self.score = None
        self.children = [None for _ in range(len(self.features))]

    def insert_key(self, key:str, score:int):
        currentNode = self
        for c in key:
            n = SBTN.features.find(c)
            if currentNode.children[n] == None:
                newNode = SBTN()
                currentNode.children[n] = newNode
            currentNode = currentNode.children[n]
        currentNode.end = True
        currentNode.score = score

    def get_key_score(self, key:str):
        currentNode = self
        for c in key:
            n = SBTN.features.find(c)
            if currentNode.children[n] == None:
                return None
            currentNode = currentNode.children[n]
        return currentNode.score if currentNode.end else None
