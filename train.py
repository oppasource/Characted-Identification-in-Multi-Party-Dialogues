import numpy as np
import model as m
import torch
from torch import nn, optim
from torch.autograd import Variable
from tqdm import tqdm

################# Variables and Parameters ################
data_path = 'data/'
save_path = 'models/'

lr = 1e-4
epochs = 10000
inp_dim = 25
hidden_dim = 64
n_classes = 8


######################## Loading Data ##########################
# Train input embeddings
train_input = np.load(data_path + 'train_input.npy')
# Train labels in form indexes from entity map
train_label_index = np.load(data_path + 'train_label_index.npy')


######################## Training ##########################
# Using gpu if available else cpu
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Creating model
# model = m.SimpleLSTM(inp_dim, hidden_dim, n_classes)
model = m.BiLSTM(inp_dim, hidden_dim, n_classes)
model = model.to(device)

# Defining loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

for epoch in range(epochs):
    # Keeping track of loss for every epoch
    total_loss = 0
    for i in tqdm(range(train_input.shape[0])):
        # Reshape numpy array to SBI (seq, batch, input_dim)
        inp = train_input[i].reshape((-1,1,25))
        # Convert numpy array to torch tensor
        inp = torch.from_numpy(inp).to(device)
        # Same for target
        target = torch.from_numpy(train_label_index[i]).to(device)

        # Forward Pass
        out = model(inp)

        # Backpropagation
        optimizer.zero_grad()
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()

        total_loss += loss
        
    avg_loss = total_loss.data.cpu().numpy() / train_input.shape[0]
    print('Epoch: ' + str(epoch+1) + ' of ' + str(epochs) +' , Loss: ' +  str(avg_loss))


    ############### Saving trained model ################
    if((epoch+1)%100 == 0):
        print('Saving model...')
        path = save_path + type(model).__name__ + '_FinalLoss_' + str(avg_loss) + '.pt'
        torch.save(model.state_dict(), path)
        print('Model saved at the path ' + path)
