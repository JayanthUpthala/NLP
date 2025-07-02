import sys
import torch
import numpy as np
def main():
 i = 4
 j = 4
 mask = torch.ones(i, j, device = 'cuda').triu_(1).bool() # try with triu_(0)
 print(mask)
 print('\n')
 attn = torch.rand((4,4)).cuda()
 print(attn)
 print('\n')
 attn_masked = attn.masked_fill(mask,-np.inf)
 print(attn_masked)

if __name__ == "__main__":
 sys.exit(int(main() or 0))
