import fire
import os
from tqdm import trange

from model import SemsimModel


def main(model_path='bart_epoch0.pth'):
    model = SemsimModel(device='cuda', src_max_length=1000, tgt_max_length=1000, alpha=0.)
    model.load_bart(model_path)

    test_src_file = open('bart/cnn_dm/test.source')

    os.makedirs(f'outputs/cnn_dm/{model_path}', exist_ok=True)
    test_hypo_file = open(f'outputs/cnn_dm/{model_path}/test.hypo', 'w')

    src_sents = [line.strip() for line in test_src_file.readlines()]
    batch_size = 16
    for i in trange(0, len(src_sents), batch_size):
        hypos = model.generate(src_sents[i: i + batch_size])

        for hypo in hypos:
            print(hypo, file=test_hypo_file)
        test_hypo_file.flush()


if __name__ == '__main__':
    fire.Fire(main)
