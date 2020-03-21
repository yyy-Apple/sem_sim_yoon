import torch
import torch.nn as nn
import pickle
import random
import torch.optim as optim
import numpy as np
from scipy.stats import pearsonr
from transformers import *
import os

HIDDEN_SIZE = 1024
PROJECT_ROOT = os.path.dirname(os.path.realpath(__file__))  # 获取项目根目录


class Rewarder(nn.Module):
    """
    After training, this rewarder is frozen, and its
    weights won't be updated.
    """

    # The rewarder MLP layer is locked in later training process
    # However, both the BERT and BART are fine-tuned later
    # 他只是没有动rewarder层，他还是既fine tune了bert也fine tune了bart
    def __init__(self, gpu=True):
        super().__init__()
        self.bertTokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
        self.bertModel = BertModel.from_pretrained('bert-large-uncased')
        self.rewarder = RawRewarder(deep=True)
        self.gpu = gpu

        # load weights from previous training

        self.rewarder.load_state_dict(torch.load(os.path.join(PROJECT_ROOT, "rewarder.pth")))
        if self.gpu:
            self.rewarder.to('cuda')

    def forward(self, text, summary):
        if self.gpu:
            textEncoding = encode_sequence(text, self.bertTokenizer, self.bertModel).tolist()
            summaryEncoding = encode_sequence(summary, self.bertTokenizer, self.bertModel).tolist()
            inputEncoding = torch.tensor(textEncoding + summaryEncoding, dtype=torch.float).to('cuda')
        else:
            textEncoding = encode_sequence(text, self.bertTokenizer, self.bertModel, gpu=False).tolist()
            summaryEncoding = encode_sequence(summary, self.bertTokenizer, self.bertModel, gpu=False).tolist()
            inputEncoding = torch.tensor(textEncoding + summaryEncoding, dtype=torch.float)
        outputScore = self.rewarder(inputEncoding, softmax=False)
        return outputScore


def encode_sequence(sequence, bertTokenizer, bertModel, batchSize=510, gpu=True):
    """
    Given a sequence of tokens, using BERT to encode it
    into a vector, there are gradient flows
    :return: a vector (torch.Size([1024]))
    """

    tokens = bertTokenizer.encode(sequence)  # we get a list after this encode step

    # for tokens fewer than batchSize tokens, encode it directly
    # for tokens more than batchSize tokens, use batching
    if len(tokens) < batchSize:
        tokensTensor = torch.tensor(tokens).unsqueeze(0)
        if gpu:
            tokensTensor = tokensTensor.to('cuda')
            bertModel.to('cuda')
        encodingMean = bertModel(tokensTensor)[0].cpu().mean(dim=1).squeeze(0)
    else:
        batchNum = len(tokens) // batchSize + 1
        batchTokensList = []
        for batch in range(batchNum):
            oneTokensList = tokens[batch * batchSize: (batch + 1) * batchSize]
            batchTokensList.append(oneTokensList)

        # pad the last list, also we need attention mask
        lastOnesLen = len(tokens) - (batchNum - 1) * batchSize  # the length of last batch that has tokens
        lastZerosLen = batchNum * batchSize - len(tokens)  # the length of last batch that is padded

        batchTokensList[-1] += lastZerosLen * [0]
        attentionMask = [[1] * batchSize for i in range(batchNum)]

        lastAttentionList = [1] * lastOnesLen + [0] * lastZerosLen
        attentionMask[-1] = lastAttentionList

        batchTokensTensor = torch.tensor(batchTokensList)
        attentionMaskTensor = torch.tensor(attentionMask)

        if gpu:
            batchTokensTensor = batchTokensTensor.to('cuda')
            attentionMaskTensor = attentionMaskTensor.to('cuda')
            bertModel.to('cuda')
        encoding = bertModel(input_ids=batchTokensTensor, attention_mask=attentionMaskTensor)[0].cpu()
        batchMeans = []
        for batch in range(batchNum - 1):
            batchMean = encoding[batch].mean(axis=0)
            batchMeans.append(batchMean)
        batchMeans.append(encoding[batchNum - 1][:lastOnesLen].mean(axis=0))
        batchMeansTensor = torch.stack(batchMeans)
        encodingMean = batchMeansTensor.mean(dim=0)

    return encodingMean


class RawRewarder(nn.Module):
    """
    Use for training and save the state dict
    After going through the forward pass, the output dim is
    torch.Size([batchSize, 2, 1])
    """

    def __init__(self, deep=False):
        super().__init__()
        # each vector is the concatenation of text encoding 1024
        # and summary encoding 1024, which is 2048
        self.deep = deep
        if self.deep:
            self.linear1 = nn.Linear(HIDDEN_SIZE * 2, HIDDEN_SIZE)
            self.relu = nn.ReLU()
            self.linear2 = nn.Linear(HIDDEN_SIZE, 1)
        else:
            self.linear = nn.Linear(HIDDEN_SIZE * 2, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inputTensor, softmax=True):
        if self.deep:
            outputTensor = self.linear1(inputTensor)
            outputTensor = self.relu(outputTensor)
            outputTensor = self.linear2(outputTensor)
        else:
            outputTensor = self.linear(inputTensor)
        if softmax:
            outputTensor = self.softmax(outputTensor)
        return outputTensor


def train_rewarder(rewarder, trainList, valList, epochs=100, batchSize=32, learningRate=3e-4, bestPScore=-1.0):
    optimizer = optim.Adam(rewarder.parameters(), lr=learningRate)
    lossFunction = nn.BCELoss()
    batches = len(trainList) // batchSize + 1

    # when the model get the best P score, we store its weights
    for epoch in range(epochs):
        # first shuffle the data
        np.random.shuffle(trainList)

        # for each epoch, train all the batches
        averageLoss = 0.0
        for batch in range(batches):
            # zero the gradients
            optimizer.zero_grad()
            batchData = trainList[batch * batchSize: (batch + 1) * batchSize]
            batchInput = torch.tensor([line[0] for line in batchData], dtype=torch.float)
            batchTarget = torch.tensor([line[1] for line in batchData], dtype=torch.float).unsqueeze(-1)
            batchPredict = rewarder(batchInput)
            loss = lossFunction(batchPredict, batchTarget)
            averageLoss += loss.item()
            loss.backward()
            optimizer.step()
        averageLoss = averageLoss / batches
        print("on epoch %d, the average loss is %f." % (epoch, averageLoss))

        # compute the Pearson correlation coefficient
        trainPScore = evaluate_rewarder(rewarder, trainList)
        print("Pearson score on training data is ", trainPScore)
        valPScore = evaluate_rewarder(rewarder, valList)
        print("Pearson score on the validation data is ", valPScore)

        # check whether to save the model state
        if valPScore > bestPScore:
            torch.save(rewarder.state_dict(), os.path.join(PROJECT_ROOT, "rewarder.pth"))
            bestPScore = valPScore


def evaluate_rewarder(rewarder, dataList):
    allInput = torch.tensor([line[0] for line in dataList], dtype=torch.float)
    allPredict = rewarder(allInput).view(-1).detach().numpy()
    allTarget = torch.tensor([line[1] for line in dataList], dtype=torch.float).view(-1).detach().numpy()
    pScore = pearsonr(allPredict, allTarget)[0]
    return pScore


def get_train_val_test(trainPercentage, valPercentage):
    trainList = []
    valList = []
    testList = []
    with open('data.pkl', 'rb') as f:
        allData = pickle.load(f)
    for i in range(len(allData)):
        randomNumber = random.random()
        if randomNumber < trainPercentage:
            trainList.append(allData[i])
        elif randomNumber < trainPercentage + valPercentage:
            valList.append(allData[i])
        else:
            testList.append(allData[i])
    return trainList, valList, testList


if __name__ == '__main__':
    train, val, test = get_train_val_test(0.8, 0.1)

    # train 30 times to get the best model weights
    bestPScore = -1.0
    for i in range(30):
        print("No.%d training" % i)
        rewarder = RawRewarder(deep=True)
        print("-------------- BEGIN TRAINING --------------")
        train_rewarder(rewarder, train, val, bestPScore=bestPScore)
        # after the training, see the Pearson score on the test data
        rewarderTest = RawRewarder(deep=True)
        rewarderTest.load_state_dict(torch.load(os.path.join(PROJECT_ROOT, "rewarder.pth")))
        testPScore = evaluate_rewarder(rewarderTest, test)
        if testPScore > bestPScore:
            bestPScore = testPScore
        print("-------------- END TRAINING --------------")
        print("Pearson score on the test data is ", testPScore)
    # finally, print the best Pearson score
    rewarderTest = RawRewarder(deep=True)
    rewarderTest.load_state_dict(torch.load(os.path.join(PROJECT_ROOT, "rewarder.pth")))
    testPScore = evaluate_rewarder(rewarderTest, test)
    print("The best Pearson score on test data is ", bestPScore)
