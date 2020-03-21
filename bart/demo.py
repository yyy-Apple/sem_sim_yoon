from bart.models.bart_utils import BARTModelWrapper
import torch


if __name__ == '__main__':
    bart = BARTModelWrapper(device="cpu")
    src_tokens = bart.encode("How are you", max_length=1024)
    tgt_tokens = bart.encode("Hi, there", max_length=1024)
    logits, extra = bart(src_tokens=src_tokens, src_lengths=torch.tensor(len(src_tokens)), prev_output_tokens=tgt_tokens)
    print(logits)
