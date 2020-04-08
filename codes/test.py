import torch
import warnings

# x = torch.rand(3, 4)
# y = torch.rand(3, 4)
#
# print((x[0] * x[1] * x[2]).sum() + (y[0] * y[1] * y[2]).sum())
#
# z = x + y
# print((z[0] * z[1] * z[2]).sum())

# y, z = torch.topk(x, k=2, dim=1)
#
# # print(x)
#
# print(y)
# print(y[:,1])
#
# print(z)

# x = torch.ones((4, 64))
# w = torch.rand(4)
# print(w[:, None] * x)
# r = torch.randn((1, 64))
# z = torch.randn((1, 64))
# y = torch.randn((400000, 64))

# from losses.lossfunction import CESmoothLossKvsAll, WeightedCESmoothLossKvsAll
#
# loss_func = CESmoothLossKvsAll(smoothing=0.15)
#
# loss_func1 = WeightedCESmoothLossKvsAll(smoothing=0.15)
#
# x = torch.randn((4, 10))
# y = torch.randint(low=0, high=2, size=(4, 10))
#
# weights = torch.ones(4)
# weights = weights / weights.sum()
#
#
# loss = loss_func(x, y)
#
# loss1 = loss_func1(x, y, weights)
#
# print(loss)
# print(loss1)



# w1 = torch.mm(x * r, y.transpose(0,1))
# w2 = torch.mm(x * z, y.transpose(0,1))
#
# w3 = torch.mm(x*(r+z) *0.5, y.transpose(0,1))
#
# w4 = (w1 + w2)/2 - w3
#
# # print(w4[0])
#
# print(w4.sum())

# w = (x, r,) + (z,)

# print(len(w))

# x = torch.rand((3, 32))
# y = torch.rand((3, 32))
#
# z = torch.cat([x, y], dim=1)
# print(z.shape)
#
# # batch_norm = torch.nn.BatchNorm1d(num_features=32)
# #
# #
# #
# # print(x.shape)
# # x = batch_norm(x)
# # print(x.shape)
# #
# # conv1 = torch.nn.Conv2d(1, 32, (3, 3), 1, 0, bias=True)
# #
# e1_embedded = x.view(-1, 1, 8, 4)
# rel_embedded = y.view(-1, 1, 8, 4)
# print(e1_embedded.shape, rel_embedded.shape)
# stacked_inputs = torch.cat([e1_embedded, rel_embedded], 3)
# print(stacked_inputs.shape)
#
# print(stacked_inputs[0][0])
# batch_norm1 = torch.nn.BatchNorm2d(num_features=1)
# w = batch_norm1(stacked_inputs)
# print(w[0][0])
#
# print(w.shape)
#
# # print(e1_embedded.shape, rel_embedded.shape, stacked_inputs.shape)
#
# z = conv1(stacked_inputs)
#
# print(z.shape)

# m = torch.nn.Conv1d(2, 2, 3, stride=1)
# # input = torch.randn(20, 16, 50)
# # output = m(input)
# #
# # print(output.shape)
# x = torch.zeros((10, 1, 12))
# y = torch.ones((10, 1, 12))
#
# z = torch.cat([x, y], dim=1)
#
# print(z.shape)

# print(z)

x = torch.rand((6, 8, 1))

indexes = [0,1,1]

x[indexes] = 0

print(x)

