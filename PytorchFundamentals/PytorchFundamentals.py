import sys
import numpy as np
import torch
def main():
 t1 = torch.zeros((4,2,3))
 print(t1.shape) # try t1.shape[0], t1.shape[1], t1.shape[2], t1,shape[-1]
 # To add an extra dimension in the beginning i.e., to make the shape(1,4,2,3)
 t2 = t1.reshape(-1,4,2,3)
 print(t2.shape)

 # To add an extra dimension in the beginning i.e., to make the shape(1,4,2,3)
 t3 = t1.view(-1,4,2,3)
 print(t3.shape)

 # To add a dimension at the end, you can use view or reshape like before
 t4 = t1.view(4,2,3,-1)
 print(t4.shape)

 # To add a dimension in the beginning or end, you can also use unsqueeze
 # dim=0 to add a dimension in the beginning, dim=-1 to add at the end
 t5 = t1.unsqueeze(dim=0)
 print(t5.shape)

 # you can squeeze to remove the dimension
 t6 = t5.squeeze(dim=0)
 print(t6.shape)
 # when you reshape, or change the view, make sure the total number of
 # elements in the tensor stays the same, e.g., if original tensor is (4,2,3)
 # you can reshape it to (4,6) as the number of elements is still 4x6=24
 # or you can reshape it to (8,3) or (1,8,3) which is still 24 elements
 t7 = t1.reshape(4,6)
 print('t7',t7.shape)
 # In reshaping, one of the dimensions can be left as -1, then it will
 # automatically figure out based on other dimensions as to what that will be
 t8 = t1.reshape(4,-1) # -1 will become 6 as t1 has 24 elements
 print('t8',t8.shape)

 # * to unpack a tuple or a list, zip operation to combine two tuples or lists
 aa = [(2,3,5),(6,7,8)]
 print('unpacked list:',*aa) # unpacks the above list aa as two tuples
 bb,cc,dd = zip(*aa) # zip(*aa)=((2,6), (3,7), (5,8)), bb=(2,6)
 print('bb:',bb)

 a = np.array([[5,3],[6,7]])
 b = torch.tensor(a, dtype=torch.int64)[None] # like unsqueeze, adds dimension
 print('After None shape:', b.shape)

 # initialize
 out1 = torch.tensor([
 [0.10, 0.50, 0.40], # correct
 [0.55, 0.20, 0.25], # wrong
 [0.60, 0.10, 0.30], # correct
 [0.15, 0.65, 0.20]]) # correct
 print('out1:',out1.shape) # [4,3]
 y = torch.tensor([1, 2, 0, 1], dtype=torch.int64) # indices, can vary 0-2
 y = y.reshape(4,1) # to match out1
 probs = out1.gather(dim=1, index=y) # dim=1 , selects index on each row
 print(probs)
 y2 = torch.tensor([[1,1], [2,0], [0,1], [1,2]], dtype=torch.int64) # targets
 probs2 = out1.gather(dim=1, index=y2)
 print(probs2)

 # taking mean along a dim, we can also use axis=1 in code below
 out1_mean = out1.mean(dim=1, keepdim=True) # keepdim, makes the output 4x1
 print(out1_mean)

 # masking
 state = torch.tensor([[0,0,0],[0,0,0],[0,0,1]])
 mask = ~(state != 0) # 0 cells will be True, 1 cells will be False
 mask = ~(state != 0)*1 # True cells will be 1, False will be 0
 mask2 = (state != 0)*-1000 # non zero cells will become -1000
 mask3 = (state > 0).float()
 print(mask3)


if __name__ == "__main__":
 sys.exit(int(main() or 0))
