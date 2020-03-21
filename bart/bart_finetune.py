import fire

from bart.models.bart import BART


BATCH_SIZE = 5
LR = 3e-5
ADAM_EPSILON = 1e-8
WEIGHT_DECAY = 0.
WARMUP_PROPORTION = 0.1


def load_data(split):
    src_file = open(f'cnn_dm/{split}.source')
    tgt_file = open(f'cnn_dm/{split}.target')
    src_texts, tgt_texts = [], []
    for src, tgt in zip(src_file.readlines(), tgt_file.readlines()):
        src_texts.append(src.strip())
        tgt_texts.append(tgt.strip())
    return src_texts, tgt_texts


def main(n_epochs=1, src_max_length=1024, tgt_max_length=1024):
    bart = BART(
        #device='cuda',
        device = "cpu",
        src_max_length=src_max_length,
        tgt_max_length=tgt_max_length)

    for split in ['train', 'dev']:
        src_texts, tgt_texts = load_data(split)
        bart.load_data(
            set_type=split,
            src_texts=src_texts,
            tgt_texts=tgt_texts)

    train_steps = n_epochs * (len(bart.train_dataset) // BATCH_SIZE + 1)
    warmup_steps = int(train_steps * WARMUP_PROPORTION)
    bart.get_optimizer(
        lr=LR,
        train_steps=train_steps,
        warmup_steps=warmup_steps,
        weight_decay=WEIGHT_DECAY,
        adam_epsilon=ADAM_EPSILON)

    bart.create_training_log(
        eval_steps=len(bart.train_dataset) // BATCH_SIZE, label='bart')
    for epoch in range(n_epochs):
        bart.train_epoch(batch_size=BATCH_SIZE)


if __name__ == '__main__':
    fire.Fire(main)

