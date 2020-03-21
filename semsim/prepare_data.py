import json
from transformers import *
import torch
import os
import pickle
import time


def encode_sequence(sequence, bertTokenizer, bertModel, batchSize=510, gpu=True):
    """
    Given a sequence of tokens, using BERT to encode it
    into a vector
    :return: a vector (torch.Size([1024]))
    """

    with torch.no_grad():
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


def create_data():
    bertTokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
    bertModel = BertModel.from_pretrained('bert-large-uncased')
    bertModel.eval()
    dataList = []
    textSummaryList = os.listdir("text_and_summary")
    counter = 1
    for textSummary in textSummaryList:
        print("deal with the %dth textSummary" % counter)
        textSummaryFile = "./text_and_summary/" + textSummary
        start = time.time()
        with open(textSummaryFile, "r", encoding="utf8") as f:
            textSummaryDict = json.loads(f.read())
            summaryList = textSummaryDict.get('summaryList')
            scoreList = textSummaryDict.get('scoreList')
            text = textSummaryDict.get('text')
            for i in range(len(summaryList) - 1):
                for j in range(i + 1, len(summaryList)):
                    textSummaryEncoding1 = encode_sequence(text, bertTokenizer, bertModel).tolist() \
                                           + encode_sequence(summaryList[i], bertTokenizer, bertModel).tolist()
                    textSummaryEncoding2 = encode_sequence(text, bertTokenizer, bertModel).tolist() \
                                           + encode_sequence(summaryList[j], bertTokenizer, bertModel).tolist()
                    pairSummary = [textSummaryEncoding1, textSummaryEncoding2]

                    if scoreList[i] > scoreList[j]:
                        preference = [1, 0]
                    elif scoreList[i] < scoreList[j]:
                        preference = [0, 1]
                    else:
                        preference = [0.5, 0.5]

                    dataList.append([pairSummary, preference])
        end = time.time()
        print("time passed ", end - start)
        counter += 1

    # write all the data to file
    dataFileName = "data.pkl"
    with open(dataFileName, 'wb') as f:
        pickle.dump(dataList, f)


if __name__ == '__main__':
    create_data()
