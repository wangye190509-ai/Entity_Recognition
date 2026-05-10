import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
from model.crf import CRF
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class BiLSTM_CRF(nn.Module):
    def __init__(self, data):
        super(BiLSTM_CRF, self).__init__()
        data.show_data_summary()
        self.embedding_dim = data.word_emb_dim
        self.hidden_dim = data.HP_hidden_dim
        self.drop = nn.Dropout(data.HP_dropout)
        self.lstm_flag = data.HP_bilstm
        if self.lstm_flag:
            self.lstm_layer = data.HP_lstm_layer
            self.lstm_hidden = data.HP_hidden_dim // 2
        else:
            self.lstm_layer = data.HP_lstm_layer
            self.lstm_hidden = data.HP_hidden_dim

        # word embedding
        self.word_embeddings = nn.Embedding(data.word_alphabet.size(), self.embedding_dim)
        if data.pretrain_word_embedding is not None:
            self.word_embeddings.weight.data.copy_(torch.from_numpy(data.pretrain_word_embedding))
        else:
            self.word_embeddings.weight.data.copy_(torch.from_numpy(self.random_embedding(data.word_alphabet.size(), self.embedding_dim)))

        # LSTM
        self.lstm = nn.LSTM(self.embedding_dim, self.lstm_hidden, num_layers=self.lstm_layer,
                            batch_first=True, bidirectional=self.lstm_flag)

        # hidden2tag
        self.hidden2tag = nn.Linear(data.HP_hidden_dim, data.label_alphabet.size() + 2)

        # CRF
        self.index2label = {}
        for ele in data.label_alphabet.instance2index:
            self.index2label[data.label_alphabet.instance2index[ele]] = ele
        self.crf = CRF(len(self.index2label), data.HP_gpu)

        # move to GPU
        self.gpu = data.HP_gpu
        if self.gpu:
            self.drop = self.drop.cuda()
            self.word_embeddings = self.word_embeddings.cuda()
            self.hidden2tag = self.hidden2tag.cuda()
            self.lstm = self.lstm.cuda()

    def random_embedding(self, vocab_size, embedding_dim):
        """
        可以用来随机初始化word embedding
        """
        pretrain_emb = np.empty([vocab_size, embedding_dim])
        scale = np.sqrt(3.0 / embedding_dim)
        for index in range(vocab_size):
            pretrain_emb[index, :] = np.random.uniform(-scale, scale, [1, embedding_dim])
        return pretrain_emb

    def _get_lstm_features(self, batch_word, batch_wordlen):
        """
        # batch_word: ([batch_size, max_sentence_length])
        """
        embeds = self.word_embeddings(batch_word)
        embeds = self.drop(embeds)
        embeds_pack = pack_padded_sequence(embeds, batch_wordlen, batch_first=True)
        out_packed, (n, c) = self.lstm(embeds_pack)
        lstm_feature, _ = pad_packed_sequence(out_packed, batch_first=True)
        lstm_feature = self.hidden2tag(lstm_feature)
        return lstm_feature

    def neg_log_likelihood(self, batch_word, mask, batch_label, batch_wordlen):
        """
        param batch_words: ([batch_size, max_sentence_length])
        param mask: ([batch_size, max_sentence_length])
        param batch_label: ([batch_size, max_sentence_length])
        param batch_wordlen: ([batch_size])
        return:
            loss : 类 tensor(3052.6426, device='cuda:0', grad_fn=<SubBackward0>)
            tag_seq: ([batch_size, max_sentence_length])
        """
        lstm_feature = self._get_lstm_features(batch_word, batch_wordlen)

        # 【关键修复】PDF 老代码用 neg_log_likelihood，实际 crf.py 用的是 neg_log_likelihood_loss
        total_loss = self.crf.neg_log_likelihood_loss(lstm_feature, mask, batch_label)

        scores, tag_seq = self.crf.viterbi_decode(lstm_feature, mask)
        return total_loss, tag_seq
        return total_loss, tag_seq

    def forward(self, batch_word, mask, batch_label, batch_wordlen):
        """
        param batch_word: ([batch_size, max_sentence_length])
        param mask: ([batch_size, max_sentence_length])
        param batch_label: ([batch_size, max_sentence_length])
        param batch_wordlen: ([batch_size])
        """
        lstm_feature = self._get_lstm_features(batch_word, batch_wordlen)
        scores, best_path = self.crf.viterbi_decode(lstm_feature, mask)
        return best_path