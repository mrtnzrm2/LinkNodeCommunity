from typing import List
# Write any import statements here
import heapq

def getMaximumEatenDishCount(N: int, D: List[int], K: int) -> int:
    class IndexedPQ:
        def __init__(self) -> None:
            self.queue = []
            self.index = {}

        def insert(self, item, priority):
            if item in self.index:
                raise ValueError("Iteam already in queue")
            
            self.index[item] = len(self.queue)
            heapq.heappush(self.queue, (priority, item))

        def isEmpty(self):
            return len(self.queue) == 0

        def pop(self):
            if self.isEmpty():
                raise IndexError("Queue is empty")
            priority, item = heapq.heappop(self.queue)
            del self.index[item]
            return item
        
        def search(self, item):
            if item not in self.index:
                return False
            return True
        
    PQ = IndexedPQ()
    k = 0
    n = 0
    for d in D:
        if k == K:
            if PQ.search(d): continue
            else:
                PQ.pop()
                PQ.insert(d, n)
                n += 1
        else:
            if PQ.search(d): continue
            else:
                PQ.insert(d, n)
                k += 1
                n += 1
    return n
  

# N = 6
# D = [1, 2, 3, 3, 2, 1]
# K = 1

# N = 6
# D = [1, 2, 3, 3, 2, 1]
# K = 2

N = 7
D = [1, 2, 1, 2, 1, 2, 1]
K = 2

print(getMaximumEatenDishCount(N, D, K))