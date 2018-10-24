import torch.nn as nn

class SimpleLSTM(nn.Module):
    def __init__(self, nIn, nHidden, nClasses):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(nIn, nHidden)
        self.lin = nn.Linear(nHidden, nClasses)

    def forward(self, input_):
        recurrent, (hidden, c) = self.lstm(input_)
        out = self.lin(recurrent)
        s,b,o = out.size()
        out = out.view(s*b,o)
        return out
    
    

# model = SimpleLSTM(25, 32, 8)
# inp = torch.randn(8,1, 25)
# model(inp)