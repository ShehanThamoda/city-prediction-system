from __future__ import unicode_literals, print_function, division
import torch
import glob
import unicodedata #provide access to unicode characters
import string
import os
import torch.nn as nn
import time
import math
import random
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
device = torch.device("cpu")
def findFiles(path): return glob.glob(path)
#file path of country list with cities
file_path = r'H:\\study\\AI\\labs\\lab 3\\names\\*.txt'
result = findFiles(file_path)
print(result)
# list of all possible letters to work in RNN
all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)
# Turn a Unicode string to plain ASCII
def unicodeToAscii(s):
 return ''.join(
 c for c in unicodedata.normalize('NFD', s)
 if unicodedata.category(c) != 'Mn'
 and c in all_letters
 )
print(unicodeToAscii('Ślusàrski'))
# Build the category_lines dictionary, a list of names per language
category_lines = {}
all_categories = []
# Read a file and split into lines
def readLines(filename):
 lines = open(filename, encoding='utf-8').read().strip().split('\n')
 return [unicodeToAscii(line) for line in lines]
# process each county file one by one
for filename in findFiles(file_path):
 category = os.path.splitext(os.path.basename(filename))[0] #get the basename of
the file and remove file extension
 all_categories.append(category) #store the each unique category
 lines = readLines(filename) #read the lines from the each file
 category_lines[category] = lines
# calculate the total number of unique categories
n_categories = len(all_categories)
# in here get the first 5 line from the France file
print(category_lines['France'][:5]) #['Marseille', 'Lyon', 'Toulouse', 'Nice', 
'Nantes']
# this function use for convert the given letter into relevant index
def letterToIndex(letter):
 return all_letters.find(letter)
# this function use for convert single letter in to representation of tensor
def letterToTensor(letter):
 tensor = torch.zeros(1, n_letters)
 tensor[0][letterToIndex(letter)] = 1
 return tensor
# this function for convert line to the tensor.
# Turn a line into a <line_length x 1 x n_letters>, or an array of one-hot letter 
vectors
def lineToTensor(line):
 tensor = torch.zeros(len(line), 1, n_letters)
 for li, letter in enumerate(line):
 tensor[li][0][letterToIndex(letter)] = 1
 return tensor
class RNN(nn.Module): #RNN inherit from nn.Module from pytorch
 #constructure method of this class
 def __init__(self, input_size, hidden_size, output_size):
 super(RNN, self).__init__() #called to parent class and initialize the 
RNN
 self.hidden_size = hidden_size
 self.i2h = nn.Linear(input_size + hidden_size, hidden_size) #linear layermaps input and hidden state to the hidden state
 self.i2o = nn.Linear(input_size + hidden_size, output_size) #linear layermaps input and hidden state to the output
 self.softmax = nn.LogSoftmax(dim=1)
 # this method for forward pass performing
 def forward(self, input, hidden):
 combined = torch.cat((input, hidden), 1)
 hidden = self.i2h(combined)
 output = self.i2o(combined)
 output = self.softmax(output)
 return output, hidden
 # this method for initialize the hidden state with zeros
 def initHidden(self):
 return torch.zeros(1, self.hidden_size)
n_hidden = 128 #number of hidden units set to 128
rnn = RNN(n_letters, n_hidden, n_categories) # create the instance of RNN
# n_categories: according to our case total categories are 20. Because we use 20 
different countries.
# this method use for identify the category (country) predicted by RNN model 
according to the output. So thats why we set output as the parameter
def categoryFromOutput(output):
 top_n, top_i = output.topk(1)
 category_i = top_i[0].item()
 return all_categories[category_i], category_i
# this method for randomly select an element from the list. So thats why set the 
list as parameter
def randomChoice(l):
 return l[random.randint(0, len(l) - 1)]
# this method provide training examples randomly for a text classification
def randomTrainingExample():
 category = randomChoice(all_categories)
 line = randomChoice(category_lines[category])
 category_tensor = torch.tensor([all_categories.index(category)], 
dtype=torch.long)
 line_tensor = lineToTensor(line)
 return category, line, category_tensor, line_tensor
# print 2 random training examples: country name and one of the city name for each 
country
for i in range(2):
 category, line, category_tensor, line_tensor = randomTrainingExample()
 print('category =', category, '/ line =', line) # category = Sri Lanka / line
= Harispattuwa
 # category = Ukrane / line = 
Kostiantynivka
# define the loss function for neural network model
criterion = nn.NLLLoss() # this NLLLoss normally use for text classification and 
for a predictions
learning_rate = 0.005 # If you set this too high, it might explode. If too low, it 
might not learn
# set the optimizer for training model with SGD optimizer
optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)
# this method for maintain training loop for the RNN.
def train(category_tensor, line_tensor):
 hidden = rnn.initHidden()
 rnn.zero_grad()
 for i in range(line_tensor.size()[0]):
 output, hidden = rnn(line_tensor[i], hidden)
 loss = criterion(output, category_tensor)
 loss.backward()
 # Add parameters' gradients to their values, multiplied by learning rate
 for p in rnn.parameters():
 p.data.add_(p.grad.data, alpha=-learning_rate)
 return output, loss.item()
n_iters = 100000
print_every = 5000
plot_every = 1000
# Keep track of losses for plotting
current_loss = 0
all_losses = []
def timeSince(since):
 now = time.time()
 s = now - since
 m = math.floor(s / 60)
 s -= m * 60
 return '%dm %ds' % (m, s)
start = time.time()
# this loop for training text classification and check the performance as well 
during the training period
for iter in range(1, n_iters + 1):
 category, line, category_tensor, line_tensor = randomTrainingExample()
 output, loss = train(category_tensor, line_tensor)
 current_loss += loss
 # Print iter number, loss, name and guess
 if iter % print_every == 0:
 guess, guess_i = categoryFromOutput(output)
 correct = ' ' if guess == category else ' (%s)' % category ✓ ✗
 print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, 
timeSince(start), loss, line, guess, correct))
 # Add current loss avg to list of losses
 if iter % plot_every == 0:
 all_losses.append(current_loss / plot_every)
 current_loss = 0
# then this graph shoes all losses
plt.figure()
plt.plot(all_losses)
# Keep track of correct guesses in a confusion matrix
confusion = torch.zeros(n_categories, n_categories)
n_confusion = 10000
# Just return an output given a line
def evaluate(line_tensor):
 hidden = rnn.initHidden()
 for i in range(line_tensor.size()[0]):
 output, hidden = rnn(line_tensor[i], hidden)
 return output
# Go through a bunch of examples and record which are correctly guessed
for i in range(n_confusion):
 category, line, category_tensor, line_tensor = randomTrainingExample()
 output = evaluate(line_tensor)
 guess, guess_i = categoryFromOutput(output)
 category_i = all_categories.index(category)
 confusion[category_i][guess_i] += 1
# Normalize by dividing every row by its sum
for i in range(n_categories):
 confusion[i] = confusion[i] / confusion[i].sum()
# Set up plot
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(confusion.numpy())
fig.colorbar(cax)
# Set up axes
ax.set_xticklabels([''] + all_categories, rotation=90)
ax.set_yticklabels([''] + all_categories)
# Force label at every tick
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
plt.show()
# this method for do the predictions based on the input line (city). In here this 
method provide 3 predictions (3 countries)
def predict(input_line, n_predictions=3):
 print('\n> %s' % input_line)
 with torch.no_grad():
 output = evaluate(lineToTensor(input_line))
 # Get top N categories
 topv, topi = output.topk(n_predictions, 1, True)
 predictions = []
 for i in range(n_predictions):
 value = topv[0][i].item()
 category_index = topi[0][i].item()
 print('(%.2f) %s' % (value, all_categories[category_index]))
 predictions.append([value, all_categories[category_index]])
predict('Paris')
#################OUTPUT#####################
# > Paris
# (-1.45) France
# (-1.52) Italy
# (-1.82) Australia