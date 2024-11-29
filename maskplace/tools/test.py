import torch
from torch.distributions import Categorical

probs = torch.tensor([4.0, 3.0, 2.0, 1.0])
pd = Categorical(probs=probs)

print(pd.probs)
print(pd)
print(probs)
ss = pd.sample()
print(ss)
print(pd.log_prob(ss))
print(torch.argmax(probs, dim=-1))

probs = torch.tensor([1,2,3,4])
pd = Categorical(probs=probs)

print(pd.probs)
print(pd)
print(probs)
ss = pd.sample()
print(ss)
print(pd.log_prob(ss))
print(torch.argmax(probs, dim=-1))


probs = torch.tensor([0.1,0.2,0.3,0.4])
pd = Categorical(probs=probs)

print(pd.probs)
print(pd)
print(probs)
ss = pd.sample()
print(ss)
print(pd.log_prob(ss))
print(torch.argmax(probs, dim=-1))
