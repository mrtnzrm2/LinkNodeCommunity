from typing import List
# Write any import statements here

def getMinCodeEntryTime(N: int, M: int, C: List[int]) -> int:
  # Write your code here
  n = N // 2
  i, time = 1, 0
  for j in C:
    k = abs(j - i)
    if k > n: time += N - k
    else: time += k

    i = j


  return time


I = 2

if I == 0:
  N = 3
  M = 3
  C = [1, 2, 3]

elif I == 1:
  N = 10
  M = 4
  C = [9, 4, 4, 8]

elif I == 2:
  N = 15
  M = 5
  C = [6, 1, 5, 10, 2]

print("time:\t", getMinCodeEntryTime(N, M, C))