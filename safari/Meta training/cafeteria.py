from typing import List
# Write any import statements here
class Node:
    # Constructor to create a new node
    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None

def compare(a , b):
    if a-b < 0: return -1
    elif a-b == 0: return 0
    else: return 1
 
# A utility function to insert
# a new node with the given key in BST
class BST:
    def __init__(self) -> None:
        self.root = None
        self.n = 0
    def insert(self, node, key):
        # If the tree is empty, return a new node
        if node is None:
            return Node(key)
    
        # Otherwise, recur down the tree
        if key < node.key:
            node.left = self.insert(node.left, key)
        elif key > node.key:
            node.right = self.insert(node.right, key)
    
        # Return the (unchanged) node pointer
        self.n += 1
        return node
    
    # Utility function to search a key in a BST
    def search(self, root : Node, key):
        # Base Cases: root is null or key is present at root
        if root is None or root.key == key:
            return root
    
        # Key is greater than root's key
        if root.key < key:
            return self.search(root.right, key)
    
        # Key is smaller than root's key
        return self.search(root.left, key)

    def floor(self, root : Node, key):
        if root is None: return None
        cmp = compare(key, root.key)
        if cmp == 0: return root
        if cmp < 0: return self.floor(root.left, key)
        t = self.floor(root.right, key)
        if t is not None: return t
        else: return root

    def ceiling(self, root : Node, key):
        if root is None: return None
        cmp = compare(key, root.key)
        if cmp == 0: return root
        if cmp > 0: return self.ceiling(root.right, key)
        t = self.ceiling(root.left, key)
        if t is not None: return t
        else: return root

def getMaxAdditionalDinersCount(N: int, K: int, M: int, S: List[int]) -> int:

    bst = BST()
    
    def add_seat(i : int):
        low = bst.floor(bst.root, i-1e-2)
        if low == None:
            low = 0
        else:
            low  = low.key
        high = bst.ceiling(bst.root, i+1e-2)
        if high == None:
            high = N+1
        else:
            high = high.key

        if high-i > 2*K + 1 and i < N - 2*K:
                S.append(i+K+1)
                bst.insert(bst.root, i+K+1)
                add_seat(i+K+1)
        if i >= N - 2*K and high == N+1 and i + K + 1 <= N:
                S.append(i+K+1)
                bst.insert(bst.root, i+K+1)
                add_seat(i+K+1)
        if i-low > 2*K + 1 and i > 2*K + 1:
                S.append(i-K-1)
                bst.insert(bst.root, i-K-1)
                add_seat(i-K-1)
        if i <= 2*K + 1 and low == 0 and i - K - 1 >= 1:
                S.append(i-K-1)
                bst.insert(bst.root, i-K-1)
                add_seat(i-K-1)
        return
    
    Sin = len(S)
    for i in range(M): bst.root = bst.insert(bst.root, S[i])

    for s in S: add_seat(s)

    return len(S)-Sin

# N = 15
# K = 2
# M = 3
# S = [11, 6, 14]

# N = 10
# K = 1
# M = 2
# S = [2, 6]

N = 13
K = 2
M = 2
S = [2, 9]

L = getMaxAdditionalDinersCount(N, K , M, S)
print(L)

