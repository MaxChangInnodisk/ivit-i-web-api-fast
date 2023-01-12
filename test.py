import numpy as np

pts = [ [1,3], [100, 200], [50, 70], [30, 140] ]

vectors = [ np.array(pts[i+1])-np.array(pts[i]) for i in range(len(pts)-1) ]
print(vectors)

num_vec = len(vectors)
sum_vec = sum(vectors)
print(num_vec, sum_vec)

avg_vec = sum_vec//num_vec
print(avg_vec)