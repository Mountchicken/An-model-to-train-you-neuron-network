import numpy as np
a=np.array([[1,2],[3,4]])
b=np.array([[1,2],[3,4]])
dic1={"b1":2,"W1":a}
dic2={"n2":3,"W2":b}
lst=[dic1,dic2]
sumd=0
for i,lt in enumerate(lst):
    sumd=sumd+np.sum(np.square(lt['W'+str(i+1)]))
print(sumd)