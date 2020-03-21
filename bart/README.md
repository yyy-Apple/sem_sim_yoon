# summarization

## Download CNN/DM
```
bash download/download_cnndm.sh
```

## Finetune
```
python bart_finetune.py [--n_epochs xxx]
```
will finetune with cnn/dm training data and the training log is in ```bart_training_logs/```.

## Generation
```
python bart_gen.py
```
will do generation and the output file is ```outputs/cnn_dm/test.hypo```.