#!/bin/python3

import math
import os
import random
import re
import sys

#
# Complete the 'timeConversion' function below.
#
# The function is expected to return a STRING.
# The function accepts STRING s as parameter.
#

def timeConversion(s):
    # Write your code here
    if "PM" in s:
        h, m, sec = s.split(":")
        h, sec = int(h), sec[:2]
        h += 12
        return f"{h}:{m}:{sec}"
    else:
        h, m, sec = s.split(":")
        h, sec = int(h), sec[:2]
        if h < 10:
            return f"0{h}:{m}:{sec}"
        elif h == 12:
            return f"00:{m}:{sec}"
        else:
            return f"{h}:{m}:{sec}"
if __name__ == '__main__':
    s = "12:28:00AM"
    result = timeConversion(s)
    print(result)
