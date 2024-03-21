from toybind import MyClass, tensorDouble
import numpy as np

m = MyClass()

a = np.random.rand(300, 300, 2)

b = tensorDouble(a)

a = np.random.rand(300, 300, 2)
c = tensorDouble(a)

m.contents = b
m.contents2 = c

s = np.array(m.sum_c())
print(s)