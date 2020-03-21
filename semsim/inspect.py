from transformers import *
import os
import time
import torch
import json

if __name__ == '__main__':
    bertTokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
    lenList = []
    textSummaryList = os.listdir("text_and_summary")
    counter = 1
    for textSummary in textSummaryList:
        print("deal with the %dth textSummary" % counter)
        textSummaryFile = "./text_and_summary/" + textSummary
        start = time.time()
        with open(textSummaryFile, "r", encoding="utf8") as f:
            textSummaryDict = json.loads(f.read())
            text = textSummaryDict.get('text')
            with torch.no_grad():
                tokens = bertTokenizer.encode(text)
                length = len(tokens)
                lenList.append(length)
        end = time.time()
        print("time passed ", end - start)
        counter += 1
    print("The longest tokens of text is ", max(lenList))  # The longest tokens of text is  2366
