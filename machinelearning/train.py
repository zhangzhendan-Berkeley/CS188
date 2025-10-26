from torch import no_grad
from torch.nn.functional import cross_entropy
from torch.utils.data import DataLoader


"""
Functions you should use.
Please avoid importing any other functions or modules.
Your code will not pass if the gradescope autograder detects any changed imports
"""
from torch import optim, tensor
from losses import regression_loss, digitclassifier_loss, languageid_loss, digitconvolution_Loss
from torch import movedim


"""
##################
### QUESTION 1 ###
##################
"""


def train_perceptron(model, dataset):
    """
    Train the perceptron until convergence.
    You can iterate through DataLoader in order to 
    retrieve all the batches you need to train on.

    Each sample in the dataloader is in the form {'x': features, 'label': label} where label
    is the item we need to predict based off of its features.
    """
    with no_grad():
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

        while True:
            error = 0
            for batch in dataloader:
                x = batch['x'].squeeze()
                label = batch['label'].item()
                y = model.get_prediction(x)
                if y != label:
                    error += 1
                    model.w.data += x * label
            if error == 0:
                break



def train_regression(model, dataset):
    """
    Trains the model.

    In order to create batches, create a DataLoader object and pass in `dataset` as well as your required 
    batch size. You can look at PerceptronModel as a guideline for how you should implement the DataLoader

    Each sample in the dataloader object will be in the form {'x': features, 'label': label} where label
    is the item we need to predict based off of its features.

    Inputs:
        model: Pytorch model to use
        dataset: a PyTorch dataset object containing data to be trained on
        
    """
    dataloader = DataLoader(dataset, batch_size= 1, shuffle=True)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    for epoch in range(300):
        for batch in dataloader:
            x = batch["x"]
            label = batch["label"]
            y = model(x)
            loss = regression_loss(y, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def train_digitclassifier(model, dataset):
    """
    Trains the model.
    """
    model.train()
    """ YOUR CODE HERE """
    dataloader = DataLoader(dataset, batch_size=200, shuffle=True)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    for epoch in range(500):
        for batch in dataloader:
            x = batch["x"]
            label = batch["label"]
            y = model(x)
            loss = digitclassifier_loss(y, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def train_languageid(model, dataset):
    """
    Trains the model.

    Note that when you iterate through dataloader, each batch will returned as its own vector in the form
    (batch_size x length of word x self.num_chars). However, in order to run multiple samples at the same time,
    get_loss() and run() expect each batch to be in the form (length of word x batch_size x self.num_chars), meaning
    that you need to switch the first two dimensions of every batch. This can be done with the movedim() function 
    as follows:

    movedim(input_vector, initial_dimension_position, final_dimension_position)

    For more information, look at the pytorch documentation of torch.movedim()
    """
    model.train()
    "*** YOUR CODE HERE ***"
    dataloader = DataLoader(dataset, batch_size=200, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    loss = 0
    for epoch in range(50):
        for batch in dataloader:
            xs = batch["x"]
            xs = movedim(xs, 1, 0)
            label = batch["label"]
            y = model(xs)
            loss = languageid_loss(y, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch}: loss = {loss.item():.4f}")

def Train_DigitConvolution(model, dataset):
    """
    Trains the model.
    """
    """ YOUR CODE HERE """
    model.train()
    "*** YOUR CODE HERE ***"
    dataloader = DataLoader(dataset, batch_size=200, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    loss = 0
    for epoch in range(30):
        for batch in dataloader:
            xs = batch["x"]
            label = batch["label"]
            y = model(xs)
            loss = digitconvolution_Loss(y, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch}: loss = {loss.item():.4f}")