perl download/gdown.pl https://drive.google.com/open?id=1goxgX-0_2Jo7cNAFrsb9BJTsrH7re6by cnn_stories_tokenized.tar.gz

tar -zxvf cnn_stories_tokenized.tar.gz

mv cnn_dm/valid.source cnn_dm/dev.source
mv cnn_dm/valid.target cnn_dm/dev.target