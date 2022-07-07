import torch


device = "cuda" if torch.cuda.is_available() else "cpu"
my_tensor = torch.tensor([[1,2,3],[4,5,6]], dtype=torch.float32,device=device,requires_grad=True)


print(my_tensor)
print(my_tensor.dtype)
print(my_tensor.device)
print(my_tensor.shape)
print(my_tensor.requires_grad)

#initalization methods
x = torch.empty(size = (3,3))
x = torch.zeros((3,3))
x= torch.rand((3,3))
x= torch.ones((3,3))
x=torch.eye(5,5)
x= torch.arange(start=0,end=5,step=1)
x = torch.linspace(start=0.1,end=1,steps=10)
x=torch.empty(size=(1,5)).normal_(mean=0,std=1)
x=torch.empty(size=(1,5)).uniform_(0,1)
x = torch.diag(torch.ones(3))

tensor = torch.arange(4)
print(tensor.bool())
print(tensor.short())
print(tensor.long())
print(tensor.half())
print(tensor.float())
print(tensor.double())

import numpy as np
np_array = np.zeros((5,5))
tensor = torch.from_numpy(np_array)
np_array_back = tensor.numpy()

##tensor math

x = torch.tensor([1,2,3])
y = torch.tensor([9,8,7])

#addition
z1 = torch.empty(3)
torch.add(x,y,out=z1)

z2 = torch.add(x,y)
z = x + y

#subtraction
z = x - y

#devision
z = torch.true_divide(x,y)

#inplace operation
t = torch.zeros(3)
t.add_(x)
t += x

#exponentiation
z = x.pow(2)
z = x**2
print(z)

#simple comparison
z = x > 0
z= x < 0

#matrix multiplication
x1 = torch.rand((2,5))
x2 = torch.rand((5,3))
x3 = torch.mm(x1,x2)
x3 = x1.mm(x2)

#matrix exponentiation
matrix_exp = torch.rand(5,5)
matrix_exp.matrix_power(3)
print(matrix_exp)

#element wise mult
z = x * y
print(z)

#dot product
z = torch.dot(x,y)
print(z)

#batch matrix multiplication
batch = 32
n = 10
m = 20
p = 30

tensor1 =torch.rand((batch,n,m))
tensor2 = torch.rand((batch,m,p))
out_bmm = torch.bmm(tensor1,tensor2) #batch,n,p

#example of broadcasting

x1 = torch.rand((5,5))
print(x1)
x2 = torch.rand((1,5))
print(x2)
z = x1 - x2
z = x1 ** x2
print(z)
#other
sum_x = torch.sum(x,dim =0)
values,indices = torch.max(x,dim=0)
values.indices = torch.min(x,dim=0)
abs_x = torch.abs(x)
z = torch.argmax(x,dim=0)
z = torch.argmin(x,dim=0)
mean_x = torch.mean(x.float(),dim=0)
z =torch.eq(x,y)
sorted_y,indices = torch.sort(y,dim=0,descending=False)

z = torch.clamp(x,min=0,max = 10 )




