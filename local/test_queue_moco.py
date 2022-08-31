import torch
ptr = 0
size_list = torch.randperm(20) + 50
print(size_list)
queue = torch.zeros(400)
num_neg = 400
i = 0
for size in size_list:
    keys = torch.linspace(1, size, size)
    if ptr+size >= num_neg:
        new_ptr = (ptr+size) % num_neg
        leave = size - new_ptr
        queue[ptr:num_neg] = keys[:leave]
        queue[:new_ptr] =  keys[leave:]
    else:
        queue[ptr:ptr+size] = keys
    i = i+1
    if i<=7:
        print(keys)
        print(queue)
    ptr = (ptr+size) % num_neg
     
    
