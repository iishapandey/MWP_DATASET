import torch
import numpy as np
from unit_kb import Unit

def add_num_to_dict(num, operand_dict, operand_mask, probability_mask, index):
    last_dict_index = torch.max((operand_mask[index] == 1).nonzero())
    operand_dict[index + 1] = operand_dict[index].clone()
    operand_dict[index + 1][:, last_dict_index + 1] = num
    operand_mask[index + 1] = operand_mask[index].clone()
    operand_mask[index + 1][:, last_dict_index + 1] = 1
    probability_mask[index + 1] = probability_mask[index].clone()
    probability_mask[index + 1][:, last_dict_index + 1] = 1
    return operand_dict, operand_mask


def execute(operator, operand_1, operand_2, operand_dict, index):
    operand_1 = operand_dict[index][:, operand_1]
    operand_2 = operand_dict[index][:, operand_2]
    add = operand_1 + operand_2
    sub = operand_1 - operand_2
    mul = operand_1 * operand_2
    div = operand_1 / operand_2
    exp = operand_1 ** operand_2
    results = torch.cat((add, sub, mul, div, exp))
    return results[operator - 1]

def execute_supervised(operator, operand_1, operand_2, operand_dict):
    operand_1 = operand_dict[range(len(operand_dict)), operand_1]
    operand_2 = operand_dict[range(len(operand_dict)), operand_2]
    #print("op1: ",operand_1)
    # print(operand_1.shape)
    #print("op2", operand_2)
    zeros = torch.zeros_like(operand_1)
    add = operand_1 + operand_2
    sub = operand_1 - operand_2
    mul = operand_1 * operand_2
    div = operand_1 / operand_2
    exp = operand_1 ** operand_2
    results = torch.stack((zeros, add, sub, mul, div, exp))
    #print(results)
    return results[operator, range(len(operand_dict))].squeeze()

def execute_unit_supervised(operator, operand_1, operand_2, operand_dict):
    operator, operand_1, operand_2 = operator.cpu(), operand_1.cpu(), operand_2.cpu()
    operand_1 = operand_dict[range(len(operand_dict)), operand_1]
    operand_2 = operand_dict[range(len(operand_dict)), operand_2]
    zeros = np.zeros(operand_1.shape, dtype=Unit)
    zeros[:] = Unit('NO')
    add = operand_1 + operand_2
    sub = operand_1 - operand_2
    mul = operand_1 * operand_2
    div = operand_1 / operand_2
    exp = operand_1 ** operand_2
    results = np.stack((zeros, add, sub, mul, div, exp))
    output = results[operator, range(len(operand_dict))].squeeze()
    partial_reward = np.where(output == Unit('None'), -1, 0)
    return partial_reward, output

def execute_single(operator, operand_1, operand_2, operand_dict):
    operand_1 = operand_dict[operand_1]
    operand_2 = operand_dict[operand_2]
    add = operand_1 + operand_2
    sub = operand_1 - operand_2
    mul = operand_1 * operand_2
    div = operand_1 / operand_2
    exp = operand_1 ** operand_2
    result = torch.tensor([0,add, sub, mul, div, exp])
    return result[operator]

def execute_supervised_beam(operator, operand_1, operand_2, operand_dict):
    operand_1 = operand_dict[torch.tensor(range(len(operand_dict))).unsqueeze(1), operand_1.long()]
    operand_2 = operand_dict[torch.tensor(range(len(operand_dict))).unsqueeze(1), operand_2.long()]
    zeros = torch.zeros_like(operand_1)
    add = operand_1 + operand_2
    sub = operand_1 - operand_2
    mul = operand_1 * operand_2
    div = operand_1 / operand_2
    exp = operand_1 ** operand_2
    results = torch.stack((zeros, add, sub, mul, div, exp))
    sizes = results.size()
    return results[operator.long(), torch.tensor(range(sizes[1])).unsqueeze(1), torch.tensor(range(sizes[2])).unsqueeze(0)]

def execute_unit_supervised_beam(operator, operand_1, operand_2, operand_dict):
    operator, operand_1, operand_2 = operator.cpu(), operand_1.cpu(), operand_2.cpu()
    operand_1 = operand_dict[torch.tensor(range(len(operand_dict))).unsqueeze(1), operand_1]
    operand_2 = operand_dict[torch.tensor(range(len(operand_dict))).unsqueeze(1), operand_2]
    zeros = np.zeros(operand_1.shape, dtype=Unit)
    zeros[:,:] = Unit('NO')
    add = operand_1 + operand_2
    sub = operand_1 - operand_2
    mul = operand_1 * operand_2
    div = operand_1 / operand_2
    exp = operand_1 ** operand_2
    results = np.stack((zeros, add, sub, mul, div, exp))
    sizes = results.shape
    output = results[operator, torch.tensor(range(sizes[1])).unsqueeze(1), torch.tensor(range(sizes[2])).unsqueeze(0)]
    partial_reward = np.where(output == Unit('None'), -1, 0)
    return partial_reward, output