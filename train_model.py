import fire
import os
import time
from model import SemsimModel

BATCH_SIZE = 32
LR = 3e-5
ADAM_EPSILON = 1e-8
WEIGHT_DECAY = 0.
WARMUP_STEPS = 500
PROJECT_ROOT = os.path.dirname(os.path.realpath(__file__)) 


def load_data(split):
    src_file = open(os.path.join(PROJECT_ROOT, f'bart/cnn_dm/{split}.source'))
    tgt_file = open(os.path.join(PROJECT_ROOT, f'bart/cnn_dm/{split}.target'))
    src_texts, tgt_texts = [], []
    for src, tgt in zip(src_file.readlines(), tgt_file.readlines()):
        src_texts.append(src.strip())
        tgt_texts.append(tgt.strip())
    return src_texts[:200000], tgt_texts[:200000]


def main(n_epochs=1, src_max_length=1024, tgt_max_length=1024):
    model = SemsimModel(
        device="cuda",
        src_max_length=src_max_length,
        tgt_max_length=tgt_max_length,
        alpha=1)

    print("Finished constructing model.")

    for split in ['train', 'dev']:
        src_texts, tgt_texts = load_data(split)
        model.load_data(
            set_type=split,
            src_texts=src_texts,
            tgt_texts=tgt_texts)

    print("Finished loading data.")

    train_steps = n_epochs * (len(model.train_dataset) // BATCH_SIZE + 1)
    # warmup_steps = int(train_steps * WARMUP_PROPORTION)
    model.get_optimizer(
        lr=LR,
        train_steps=train_steps,
        warmup_steps=WARMUP_STEPS,
        weight_decay=WEIGHT_DECAY,
        adam_epsilon=ADAM_EPSILON)

    model.create_training_log(
        eval_steps=len(model.train_dataset) // BATCH_SIZE, label='bart')

    for epoch in range(n_epochs):
        model.train_epoch(batch_size=BATCH_SIZE)
        model.save_bart(f'bart_epoch{epoch}.pth')
    exit()


    rouge_best = 0
    for epoch in range(n_epochs):
        start = time.time()
        model.train_epoch(batch_size=BATCH_SIZE)
        rouge1, rouge2, rougel = model.evaluate("./bart/cnn_dm/dev.source", "./bart/cnn_dm/dev.predict",
                                                "./bart/cnn_dm/dev.target")
        rouge = (rouge1 + rouge2 + rougel) / 3
        end = time.time()
        print("On epoch %d, the average Rouge is %f." % (epoch, rouge))
        print("Time passed %d" % (end - start))

        if rouge > rouge_best:
            model.save_bart("bart.pth")
            model.save_bert("bert.pth")
            rouge_best = rouge

    # after the training, perform on test data
    model_test = SemsimModel(
        device="cuda",
        src_max_length=src_max_length,
        tgt_max_length=tgt_max_length,
        alpha=1)

    model_test.load_bart("bart.pth")
    model_test.load_bert("bert.pth")
    rouge1, rouge2, rougel = model.evaluate("./bart/cnn_dm/test.source", "./bart/cnn_dm/test.predict",
                                            "./bart/cnn_dm/test.target")
    with open("final_result.txt", "w", encoding="utf8") as f:
        f.write("Rouge-1: " + str(rouge1))
        f.write("\n")
        f.write("Rouge-2: " + str(rouge2))
        f.write("\n")
        f.write("Rouge-l: " + str(rougel))


if __name__ == '__main__':
    fire.Fire(main)

