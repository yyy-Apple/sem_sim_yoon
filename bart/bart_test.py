from fairseq.data.data_utils import collate_tokens
from bart.models.bart import BART
from bart.models.bart_utils import BARTModelWrapper
import torch


if __name__ == '__main__':
    bart = torch.hub.load('pytorch/fairseq', 'bart.large.cnn')
    src_tokens = bart.encode("How are you")
    tgt_tokens = bart.encode("Hi, there")
    print(src_tokens)
    print(tgt_tokens)
    print(bart.decode(src_tokens))
    print(bart.decode(tgt_tokens))
    print(bart.decode(torch.tensor([0, 30, 67, 30, 30, 30])))
