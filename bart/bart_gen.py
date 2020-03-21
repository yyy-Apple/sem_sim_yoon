import fire
import os

from bart.models.bart import BART


def main():
    bart = BART(device='cuda', src_max_length=1000, tgt_max_length=1000)

    test_src_file = open('cnn_dm/test.source')

    os.makedirs('outputs/cnn_dm', exist_ok=True)
    test_hypo_file = open('outputs/cnn_dm/test.hypo', 'w')

    for src in test_src_file.readlines():
        hypo_sum = bart.generate(src.strip())

        print(hypo_sum)
        print(hypo_sum, file=test_hypo_file)
        test_hypo_file.flush()


if __name__ == '__main__':
    fire.Fire(main)