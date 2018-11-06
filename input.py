import model as m
import torch
from torch import nn, optim
from torch.autograd import Variable

from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models.keyedvectors import KeyedVectors

import re
import string
import numpy as np
import os.path
import nltk
import pdb

entities_dict = {0: "Rachel Green", 1: "Ross Geller", 2: "Chandler Bing", 3: "Monica Geller", 
				4: "Joey Tribbiani", 5: "Phoebe Buffay", 6: "Others", 7: "None"}


################ For getting embeddings ##################
if os.path.isfile('pretrained_embeds/gensim_glove_vectors.txt'):
    glove_model = KeyedVectors.load_word2vec_format("pretrained_embeds/gensim_glove_vectors.txt", binary=False)
else:
    glove2word2vec(glove_input_file="pretrained_embeds/glove.twitter.27B.25d.txt", word2vec_output_file="pretrained_embeds/gensim_glove_vectors.txt")
    glove_model = KeyedVectors.load_word2vec_format("pretrained_embeds/gensim_glove_vectors.txt", binary=False)

def get_embed(word):
    # Case folding
    word = word.lower()
    try:
        return (glove_model.get_vector(word))
    except:
        return (glove_model.get_vector('unk'))
###########################################################


####################### Loading Saved Model ####################
inp_dim = 25
hidden_dim = 64
n_classes = 8

save_path = 'models/'
data_path = 'data/'
model = 'SimpleLSTM_FinalLoss_0.11093811956997775.pt'

# Using gpu if available else cpu
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

mo = m.SimpleLSTM(inp_dim, hidden_dim, n_classes)
mo.load_state_dict(torch.load(save_path + model))
#################################################################


# Keep asking for different input 
while(True):
	print('\n\nChoose a speaker\n')

	for e in entities_dict.items():
		print(e)

	# Getting speaker input
	try:
		inp = int(input())
	except:
		continue

	# Getting speaker embedding
	temp_speaker = []
	for s in entities_dict[inp].split():
	    temp_speaker.append(get_embed(s))
	temp_speaker = np.asarray(temp_speaker)
	temp_speaker = np.mean(temp_speaker, axis=0).reshape(1,-1)

	# Getting spoken utterance
	utterance = input('Enter the utterance: ')
	utterance_tokens = nltk.word_tokenize(utterance)

	# Getting embeddings for tokens 
	temp = []
	for t in utterance_tokens:
	    temp.append(get_embed(t))
	temp = np.asarray(temp)

	# Concatinating speaker and utterance
	final_input = np.vstack((temp_speaker, temp))


	# Getting output from trained model
	final_input = torch.from_numpy(final_input.reshape((-1,1,25))).to(device)
	out = mo(final_input)
	out = torch.max(out,1)[1]
	out = out[1:]

	# Printing output beautifully
	print('--------------------------------------------------------')
	for i in range(len(utterance_tokens)):
		if out[i].item() != 7:
			print('{:<15s}'.format(utterance_tokens[i]) + '{:<10s}'.format(entities_dict[out[i].item()]))
		else:
			print('{:<15s}'.format(utterance_tokens[i]) + '{:<10s}'.format('-'))
	print('--------------------------------------------------------')
