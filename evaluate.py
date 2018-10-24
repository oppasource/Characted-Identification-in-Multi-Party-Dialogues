import numpy as np
import model as m
import torch
from torch import nn, optim
from torch.autograd import Variable

inp_dim = 25
hidden_dim = 64
n_classes = 8

save_path = 'models/'
data_path = 'data/'

######################## Loading Data ##########################
# Train input embeddings
train_input = np.load(data_path + 'train_input.npy')
# Train labels in form indexes from entity map
train_label_index = np.load(data_path + 'train_label_index.npy')

# Test input embeddings
test_input = np.load(data_path + 'test_input.npy')
# Test labels in form indexes from entity map
test_label_index = np.load(data_path + 'test_label_index.npy')


####################### Loading Saved Model ####################
# Using gpu if available else cpu
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

mo = m.SimpleLSTM(inp_dim, hidden_dim, n_classes)
mo.load_state_dict(torch.load(save_path + 'SimpleLSTM_FinalLoss_0.21003304342390275.pt'))

################## Forward Pass for test data #################
# Code is just for evaluations
total_tokens = 0
total_seq = 0

correct_tokens = 0
correct_seq = 0

i = 155

for i in range(test_input.shape[0]):
    # Prepare the input
    inp = torch.from_numpy(test_input[i].reshape((-1,1,25))).to(device)
    truth = torch.from_numpy(test_label_index[i])
    # Get output from trained model
    out = mo(inp)
    out = torch.max(out,1)[1]
    
    # Following code is to calculate accuracy seperately for sequence and token entities
    check = torch.eq(truth,out)
    seq_len = check.size()[0]
    correct_tokens_temp = torch.sum(check).item()
    correct_tokens += correct_tokens_temp
    
    if (seq_len == correct_tokens_temp):
        correct_seq += 1
    
    total_tokens += seq_len
    total_seq += 1

    
token_accuracy = correct_tokens / total_tokens
seq_accuracy = correct_seq / total_seq
print('Accuracy considering one entity at a time: ' + str(token_accuracy))
print('Accuracy considering one whole sequence at a time: ' + str(seq_accuracy))
