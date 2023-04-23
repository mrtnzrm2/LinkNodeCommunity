# Insert path ---
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))


from various.omega import Omega

C1 = {
    "one" : [1, 2],
    "two" : [1, 3]
}

C2 = {
    "one" : [1, 3],
    "two" : [1, 2, 3],
    "three" : [2, 3]
}

A  = Omega(C1, C2).omega_score
print(C1.items())
print(A)