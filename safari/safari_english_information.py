import numpy as np
import pandas as pd
import os

data = pd.read_csv("../CSV/Others/english_letter_counts.csv")

# data["Pr"] = data["Count"] / data["Count"].sum()
# data["I"] = np.ceil(-np.log2(data["Pr"]))

# print(data)

# A Huffman Tree Node 
import heapq 
  
  
class node: 
    def __init__(self, freq, symbol, left=None, right=None): 
        # frequency of symbol 
        self.freq = freq 
  
        # symbol name (character) 
        self.symbol = symbol 
  
        # node left of current node 
        self.left = left 
  
        # node right of current node 
        self.right = right 
  
        # tree direction (0/1) 
        self.huff = '' 
  
    def __lt__(self, nxt): 
        return self.freq < nxt.freq 
  
  
# utility function to print huffman 
# codes for all symbols in the newly 
# created Huffman tree 
def printNodes(node, val=''): 
  
    # huffman code for current node 
    newVal = val + str(node.huff) 
  
    # if node is not an edge node 
    # then traverse inside it 
    if(node.left): 
        printNodes(node.left, newVal) 
    if(node.right): 
        printNodes(node.right, newVal) 
  
        # if node is edge node then 
        # display its huffman code 
    if(not node.left and not node.right): 
        print(f"{node.symbol} -> {newVal}") 

def GetCode(node, codebook, val=''): 
    
    # huffman code for current node 
    newVal = val + str(node.huff) 
  
    # if node is not an edge node 
    # then traverse inside it 
    if(node.left): 
        GetCode(node.left, codebook, newVal) 
    if(node.right): 
        GetCode(node.right, codebook, newVal) 
  
        # if node is edge node then 
        # display its huffman code 
    if(not node.left and not node.right): 
        # print(f"{node.symbol} -> {newVal}")
        codebook[node.symbol] = newVal
  
  
# characters for huffman tree 
chars = [l for l in data["Letter"]]
chars.append(" ")
# frequency of characters 
freq = [f for f in data["Count"]]
freq.append(data["Count"].iloc[0] * 20 / 12.7)

# freq = np.random.permutation(freq)
  
# list containing unused nodes 
nodes = [] 
  
# converting characters and frequencies 
# into huffman tree nodes 
for x in range(len(chars)): 
    heapq.heappush(nodes, node(freq[x], chars[x])) 
  
while len(nodes) > 1: 
  
    # sort all the nodes in ascending order 
    # based on their frequency 
    left = heapq.heappop(nodes) 
    right = heapq.heappop(nodes) 
  
    # assign directional value to these nodes 
    left.huff = 0
    right.huff = 1
  
    # combine the 2 smallest nodes to create 
    # new node as their parent 
    newNode = node(left.freq+right.freq, left.symbol+right.symbol, left, right) 
  
    heapq.heappush(nodes, newNode) 
  
# # Huffman Tree is ready! 
# printNodes(nodes[0]) 
    
text = "Once when I was six years old I saw a magnificent picture in a book, called True Stories from Nature, about the primeval forest. It was a picture of a boa constrictor in the act of swallowing an animal. Here is a copy of the drawing."
text = [t.upper() for t in text if t in chars]

codebook = {}
GetCode(nodes[0], codebook)

huff_text = ""
for t in text:
    huff_text += codebook[t]

print(huff_text)

def count_chars(txt):
	result = 0
	for char in txt:
		result += 1     # same as result = result + 1
	return result

print(count_chars(huff_text))