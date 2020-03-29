import os
from bart.models.bart import BART
from semsim.rewarder import Rewarder
import torch
import torch.nn as nn
import random
from tqdm import tqdm, trange
import torch.nn.functional as F
from fairseq.data.data_utils import collate_tokens
from fairseq.sequence_generator import SequenceGenerator
from rouge import FilesRouge

from transformers import AdamW, get_linear_schedule_with_warmup

LIL_BATCH_SIZE = 1


class SemsimModel(BART):
    def __init__(self, device, src_max_length, tgt_max_length, alpha, rewarder_gpu_no=0):
        # alpha is used to control the ratio of those two kinds of loss
        super().__init__(device, src_max_length, tgt_max_length)
        self.alpha = alpha
        self.rewarder = Rewarder(os.path.join('semsim', 'trained_models', 'sample.model'))

    def get_optimizer(self, lr, train_steps, warmup_steps,
                      weight_decay, adam_epsilon):
        no_decay = ["bias", "LayerNorm.weight"]
        # fine-tune both BERT and BART
        optimizer_grouped_parameters = [
            {"params": [p for n, p in self._bart.named_parameters()
                        if not any(nd in n for nd in no_decay)],
             "weight_decay": weight_decay},
            {"params": [p for n, p in self._bart.named_parameters()
                        if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0}]
        self._optimizer = AdamW(
            optimizer_grouped_parameters, lr=lr, eps=adam_epsilon)
        self._lr_scheduler = get_linear_schedule_with_warmup(
            self._optimizer, num_warmup_steps=warmup_steps,
            num_training_steps=train_steps)

    def save_bert(self, path):
        torch.save(self.rewarder.bertModel.state_dict(), path)
        print(f'BERT saved in {path}.')

    def load_bert(self, path):
        self.rewarder.bertModel.load_state_dict(torch.load(path, map_location=self._device))

    def save_bart(self, path):
        torch.save(self._bart.state_dict(), path)
        print(f'BART saved in {path}.')

    def load_bart(self, path):
        self._bart.load_state_dict(torch.load(path, map_location=self._device))

    def train_epoch(self, batch_size):
        assert 'train' in self._dataset

        random.shuffle(self._dataset['train'])
        for i in trange(0, len(self._dataset['train']), batch_size,
                        desc='BART Training'):
            self._bart.set_mode('train')
            self._bart.train()

            batch = self._dataset['train'][i:i + batch_size]

            self._optimizer.zero_grad()

            for j in range(0, len(batch), LIL_BATCH_SIZE):
                lil_batch = batch[j:j + LIL_BATCH_SIZE]

                src_lengths = torch.tensor(
                    [len(t.src_tokens) for t in lil_batch])
                src_tokens = collate_tokens(
                    [t.src_tokens for t in lil_batch],
                    pad_idx=self._bart.dictionary.pad())
                tgt_tokens = collate_tokens(
                    [t.tgt_tokens for t in lil_batch],
                    pad_idx=self._bart.dictionary.pad())

                loss = self._get_seq2seq_loss(
                    src_lengths=src_lengths,
                    src_tokens=src_tokens,
                    tgt_tokens=tgt_tokens)

                loss = loss * len(lil_batch) / batch_size
                loss.backward()

            self._optimizer.step()
            self._lr_scheduler.step()

            self._global_step += 1

    #def evaluate(self, source_path, predict_path, target_path):
    #    self._bart.set_mode('train')
    #    self._bart.eval()

    #    predict_list = []
    #    with open(source_path, "r", encoding="utf8") as f:
    #        for line in f.readlines():
    #            predict_list.append(self.generate(line))

    #    with open(predict_path, "w", encoding="utf8") as f:
    #        for prediction in predict_list:
    #            f.write(prediction)
    #            f.write("\n")

    #    files_rouge = FilesRouge()
    #    scores = files_rouge.get_scores(predict_path, target_path, avg=True)
    #    return scores.get('rouge-1').get('f'), scores.get('rouge-2').get('f'), scores.get('rouge-l').get('f')

    def _get_semsim_score(self, logits, tgt_tokens):
        try:
            with torch.no_grad():
                predict_text = self._bart.decode(logits.argmax(dim=2).squeeze(dim=0))
                # tgt_tokens = tgt_tokens.to(logits.device)
                tgt_text = self._bart.decode(tgt_tokens.squeeze(dim=0))
                score = self.rewarder(predict_text, tgt_text)

                #print(f'Predict: {predict_text}')
                #print(f'Target: {tgt_text}')
                #print(f'Score: {score}', type(score))

                return score
        except:
            return 0.

    def _get_seq2seq_loss(self, src_lengths, src_tokens, tgt_tokens):
        logits, extra = self._bart(
            src_tokens=src_tokens,
            src_lengths=src_lengths,
            prev_output_tokens=tgt_tokens)

        tgt_tokens = tgt_tokens.to(logits.device)
        semsim_score = self._get_semsim_score(logits, tgt_tokens)

        # use reinforcement learning to assign semantic similarity reward
        log_probs = F.log_softmax(logits, dim=2)
        values, _ = torch.max(log_probs, dim=2)
        # print('values.shape:', values.shape)
        # print('sum(values):', torch.sum(values))
        semsim_loss = -torch.mean(values[..., 5:]) * semsim_score
        # print('semsim_loss: ', semsim_loss, values.shape, torch.sum(values), semsim_score)

        smooth_loss = torch.mean(-log_probs[..., 2:]) # remove <pad>

        # Shift so that tokens < n predict n
        shift_logits = logits[:, :-1].contiguous()
        shift_labels = tgt_tokens[:, 1:].contiguous()

        # Flatten the tokens
        criterion = torch.nn.CrossEntropyLoss(
            ignore_index=self._bart.dictionary.pad())
        ml_loss = criterion(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        # loss = ml_loss #- self.alpha * semsim_loss
        #print(ml_loss.item() , smooth_loss.item(), semsim_loss.item())
        loss = 0.8 * ml_loss + 0.1 * smooth_loss + 0.1 * semsim_loss

        return loss

