
# coding: utf-8

# In[1]:


import os
import sys
import tqdm
import math
from collections import Counter
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import nltk
import code



# In[2]:


def load_data(in_file):
    cn = []
    en = []
    num_examples = 0
    with open(in_file, 'r') as f:
        for line in f:
            line = line.strip().split("\t")
            
            en.append(["BOS"] + nltk.word_tokenize(line[0]) + ["EOS"])
            # split chinese sentence into characters
            cn.append(["BOS"] + [c for c in line[1]] + ["EOS"])
    return en, cn

train_file = "nmt/en-cn/train.txt"
dev_file = "nmt/en-cn/dev.txt"
train_en, train_cn = load_data(train_file)
dev_en, dev_cn = load_data(dev_file)


# 构建单词表

# In[3]:


def build_dict(sentences, max_words=50000):
    word_count = Counter()
    for sentence in sentences:
        for s in sentence:
            word_count[s] += 1
    ls = word_count.most_common(max_words)
    total_words = len(ls) + 1
    word_dict = {w[0]: index+2 for index, w in enumerate(ls)}
    word_dict["UNK"] = 0
    word_dict["PAD"] = 0
    return word_dict, total_words

en_dict, en_total_words = build_dict(train_en)
cn_dict, cn_total_words = build_dict(train_cn)
inv_en_dict = {v: k for k, v in en_dict.items()}
inv_cn_dict = {v: k for k, v in cn_dict.items()}


# 把单词全部转变成数字

# In[4]:


def encode(en_sentences, cn_sentences, en_dict, cn_dict, sort_by_len=True):
    '''
        Encode the sequences. 
    '''
    length = len(en_sentences)
    out_en_sentences = [[en_dict.get(w, 0) for w in sent] for sent in en_sentences]
    out_cn_sentences = [[cn_dict.get(w, 0) for w in sent] for sent in cn_sentences]

    # sort sentences by english lengths
    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))
       
    if sort_by_len:
        sorted_index = len_argsort(out_en_sentences)
        out_en_sentences = [out_en_sentences[i] for i in sorted_index]
        out_cn_sentences = [out_cn_sentences[i] for i in sorted_index]
        
    return out_en_sentences, out_cn_sentences

train_en, train_cn = encode(train_en, train_cn, en_dict, cn_dict)
dev_en, dev_cn = encode(dev_en, dev_cn, en_dict, cn_dict)


# 把全部句子分成batch

# In[5]:


def get_minibatches(n, minibatch_size, shuffle=False):
    idx_list = np.arange(0, n, minibatch_size)
    if shuffle:
        np.random.shuffle(idx_list)
    minibatches = []
    for idx in idx_list:
        minibatches.append(np.arange(idx, min(idx + minibatch_size, n)))
    return minibatches

def prepare_data(seqs):
    lengths = [len(seq) for seq in seqs]
    n_samples = len(seqs)
    max_len = np.max(lengths)

    x = np.zeros((n_samples, max_len)).astype('int32')
#     x_mask = np.zeros((n_samples, max_len)).astype('float32')
    x_lengths = np.array(lengths).astype("int32")
    for idx, seq in enumerate(seqs):
        x[idx, :lengths[idx]] = seq
#         x_mask[idx, :lengths[idx]] = 1.0
    return x, x_lengths #x_mask

def gen_examples(en_sentences, cn_sentences, batch_size):
    minibatches = get_minibatches(len(en_sentences), batch_size)
    all_ex = []
    for minibatch in minibatches:
        mb_en_sentences = [en_sentences[t] for t in minibatch]
        mb_cn_sentences = [cn_sentences[t] for t in minibatch]
        mb_x, mb_x_len = prepare_data(mb_en_sentences)
        mb_y, mb_y_len = prepare_data(mb_cn_sentences)
        all_ex.append((mb_x, mb_x_len, mb_y, mb_y_len))
    return all_ex

batch_size = 64
train_data = gen_examples([batch[::-1] for batch in train_en], train_cn, batch_size)
dev_data = gen_examples(dev_en, dev_cn, batch_size)


# In[6]:


train_en[0]


# In[7]:


[inv_en_dict[idx] for idx in train_en[0]]


# 数据全部处理完成，现在我们开始构建seq2seq模型

# In[8]:


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, enc_hidden_size, dec_hidden_size, dropout=0.2):
        super(Encoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, enc_hidden_size, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(enc_hidden_size * 2, dec_hidden_size)

    def forward(self, x, lengths):
        sorted_len, sorted_idx = lengths.sort(0, descending=True)
        x_sorted = x[sorted_idx.long()]
        embedded = self.dropout(self.embed(x_sorted))
        
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, sorted_len.long().cpu().data.numpy(), batch_first=True)
        packed_out, hid = self.rnn(packed_embedded)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        _, original_idx = sorted_idx.sort(0, descending=False)
        out = out[original_idx.long()].contiguous()
        
        hid = torch.cat([hid[-2], hid[-1]], dim=1)
        hid = torch.tanh(self.fc(hid)).unsqueeze(0)

        # code.interact(local=locals())
        
        return out, hid


# In[9]:


class Attention(nn.Module):
    def __init__(self, enc_hidden_size, dec_hidden_size):
        super(Attention, self).__init__()

        self.enc_hidden_size = enc_hidden_size
        self.dec_hidden_size = dec_hidden_size

        self.linear_in = nn.Linear(enc_hidden_size*2, dec_hidden_size, bias=False)
        # self.bilinear_attn = nn.Bilinear(enc_hidden_size, dec_hidden_size, 1, bias=False)
        self.linear_out = nn.Linear(enc_hidden_size*2 + dec_hidden_size, dec_hidden_size)
        
    def forward(self, output, context, mask):
        # output: batch_size, output_len, dec_hidden_size
        # context: batch_size, context_len, enc_hidden_size
    
        batch_size = output.size(0)
        output_len = output.size(1)
        input_len = context.size(1)
        
        context_in = self.linear_in(context.view(batch_size*input_len, -1)).view(                batch_size, input_len, -1) # batch_size, output_len, dec_hidden_size
        # code.interact(local=locals())
        attn = torch.bmm(output, context_in.transpose(1,2)) # batch_size, output_len, context_len

        
        attn.data.masked_fill(mask, -1e6)

        attn = F.softmax(attn, dim=2) # batch_size, output_len, context_len

        context = torch.bmm(attn, context) # batch_size, output_len, enc_hidden_size
        
        output = torch.cat((context, output), dim=2) # batch_size, output_len, hidden_size*2

        
        output = output.view(batch_size*output_len, -1)
        output = F.tanh(self.linear_out(output))
        output = output.view(batch_size, output_len, -1)
        return output, attn



# In[10]:


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, enc_hidden_size, dec_hidden_size, dropout=0.2):
        super(Decoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.attention = Attention(enc_hidden_size, dec_hidden_size)
        self.rnn = nn.GRU(embed_size, hidden_size, batch_first=True)
        self.out = nn.Linear(dec_hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def create_mask(self, x_len, y_len):
        device = x_len.device
        max_x_len = x_len.max()
        max_y_len = y_len.max()
        x_mask = torch.arange(max_x_len, device=x_len.device)[None, :] < x_len[:, None]
        y_mask = torch.arange(max_y_len, device=x_len.device)[None, :] < y_len[:, None]
        mask = (1 - x_mask[:, :, None] * y_mask[:, None, :]).byte()
        return mask
        
        
    def forward(self, ctx, ctx_lengths, y, y_lengths, hid):
        sorted_len, sorted_idx = y_lengths.sort(0, descending=True)
        y_sorted = y[sorted_idx.long()]
        hid = hid[:, sorted_idx.long()]

        y_sorted = self.dropout(self.embed(y_sorted)) # batch_size, output_length, embed_size

        packed_seq = nn.utils.rnn.pack_padded_sequence(y_sorted, sorted_len.long().cpu().data.numpy(), batch_first=True)
        out, hid = self.rnn(packed_seq, hid)
        unpacked, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        _, original_idx = sorted_idx.sort(0, descending=False)
        output_seq = unpacked[original_idx.long()].contiguous()

        mask = self.create_mask(y_lengths, ctx_lengths)

        # code.interact(local=locals())
        output, attn = self.attention(output_seq, ctx, mask)
        output = F.log_softmax(self.out(output), -1)
        
        return output
    


# In[11]:


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, x, x_lengths, y, y_lengths):
        encoder_out, hid = self.encoder(x, x_lengths)
        output = self.decoder(ctx=encoder_out, 
                    ctx_lengths=x_lengths,
                    y=y,
                    y_lengths=y_lengths,
                    hid=hid)
        return output


# 训练

# In[12]:


class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, input, target, mask):
        input = input.contiguous().view(-1, input.size(2))
        target = target.contiguous().view(-1, 1)
        mask = mask.contiguous().view(-1, 1)
        output = -input.gather(1, target) * mask
        output = torch.sum(output) / torch.sum(mask)

        return output


# In[39]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

en_vocab_size = len(en_dict)
cn_vocab_size = len(cn_dict)
embed_size = hidden_size = 100
encoder = Encoder(vocab_size=en_vocab_size, 
                  embed_size=embed_size, 
                  enc_hidden_size=hidden_size,
                  dec_hidden_size=hidden_size)
decoder = Decoder(vocab_size=cn_vocab_size, 
                  embed_size=embed_size, 
                  enc_hidden_size=hidden_size,
                  dec_hidden_size=hidden_size)
model = Seq2Seq(encoder, decoder)
model = model.to(device)
crit = LanguageModelCriterion().to(device)
optimizer = torch.optim.Adam(model.parameters())


# In[14]:


len(cn_dict)


# In[40]:


num_epochs = 20
total_num_words = total_loss = 0.
for epoch in range(num_epochs):
    for it, (mb_x, mb_x_lengths, mb_y, mb_y_lengths) in enumerate(train_data):
#         break
        mb_x = torch.from_numpy(mb_x).long().to(device)
        mb_x_lengths = torch.from_numpy(mb_x_lengths).long().to(device)
        mb_input = torch.from_numpy(mb_y[:,:-1]).long().to(device)
        mb_out = torch.from_numpy(mb_y[:, 1:]).long().to(device)
        mb_y_lengths = torch.from_numpy(mb_y_lengths-1).long().to(device)
#         print(mb_input.shape)
#         print(mb_y_lengths)
#         print(mb_y_lengths.max())
        mb_y_lengths[mb_y_lengths <= 0] = 1
        
        mb_pred = model(mb_x, mb_x_lengths, mb_input, mb_y_lengths)
        
        mb_out_mask = torch.arange(mb_y_lengths.max().item(), device=device)[None, :] < mb_y_lengths[:, None]
        mb_out_mask = mb_out_mask.float()
        # code.interact(local=locals())
        loss = crit(mb_pred, mb_out, mb_out_mask)
        
        num_words = torch.sum(mb_y_lengths).item()
        total_loss += loss.item() * num_words
        total_num_words += num_words

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
        
        if it % 100 == 0:
            print("epoch", epoch, "iteration", it, "loss", loss.item())
            


# In[41]:


mb_y_lengths

