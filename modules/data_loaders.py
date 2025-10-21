import torch
import torch.nn as nn
import math
import numpy as np
import os
import re
from collections import defaultdict
import operator
import struct
import sys
from typing import Tuple


def read_file(path):
    with open(path, "r") as fin:
            data = fin.read()[::2]
    return data

def read_words(ptb_path: str, fname: str):
        raw = read_file(ptb_path + fname)
        words = []
        for token in raw.split():
            parts = token.split('_')
            words.extend(parts)
            words.append('<eos>')
        return words

def load_PTB_word(ptb_path: str, fname_train: str, fname_val: str, fname_test: str
                 ) -> Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor]:
    """
    Читает файлы PTB (word-level split by '_' as in original) и возвращает
    train, val, test как 1D torch.LongTensor индексов слов.
    """

    train_words = read_words(ptb_path, fname_train)
    val_words = read_words(ptb_path, fname_val)
    test_words = read_words(ptb_path, fname_test)

    vocab = list(set(train_words + val_words + test_words))
    vocab_size = len(vocab)
    print("vocab_size =", vocab_size)

    word_to_id = {w: i for i, w in enumerate(vocab)}
    id_to_word = {i: w for i, w in enumerate(vocab)}

    train_idx = [word_to_id[w] for w in train_words]
    val_idx = [word_to_id[w] for w in val_words]
    test_idx = [word_to_id[w] for w in test_words]

    return (torch.tensor(train_idx, dtype=torch.long),
            torch.tensor(val_idx, dtype=torch.long),
            torch.tensor(test_idx, dtype=torch.long),
            word_to_id, 
            id_to_word)

class PTB_word:
    """
    Аналог оригинального класса, но на torch.Tensor.
    Принимает 1D torch.LongTensor (последовательность индексов).
    Поведение:
      - усечение до кратного batch_size,
      - reshape -> (batch_size, -1), затем transpose -> (num_steps, batch_size)
      - get_next_batch возвращает (batch, seq_len) входы и цели.
    """
    def __init__(self, data: torch.LongTensor, length: int, batch_size: int):
        """
        data: 1D torch.LongTensor
        length: длина последовательности (seq_len)
        batch_size: batch size
        """
        if not torch.is_tensor(data):
            raise TypeError("data must be a torch.Tensor (1D) of long/int")

        if data.dim() != 1:
            raise ValueError("data must be 1D tensor")

        self.length = int(length)
        self.batch_size = int(batch_size)

        self.raw_len = data.size(0)

        n_full = (self.raw_len // self.batch_size) * self.batch_size
        truncated = data[:n_full].clone()

        cols = n_full // self.batch_size
        reshaped = truncated.view(self.batch_size, cols).t().contiguous()

        self.data = reshaped
        self.batch_ind = 0

        self.num_batches = int(math.ceil(self.raw_len / float(self.batch_size * self.length)))

    def new_epoch(self):
        self.batch_ind = 0

    def to_first_batch(self):
        self.batch_ind = 0

    def get_next_batch(self):
        """
        Возвращает (inputs, targets), оба shape (batch_size, seq_len), dtype long.
        Поведение: берёт seq_len = min(self.length, remaining_rows - 1),
        затем slice rows [batch_ind : batch_ind + seq_len + 1] и возвращает
        batch[:-1].T (входы) и batch[1:].T (цели) — как в оригинале.
        """
        n_rows = self.data.size(0)
        seq_len = min(self.length, max(0, n_rows - self.batch_ind - 1))
        if seq_len <= 0:
            return (torch.empty((self.batch_size, 0), dtype=torch.long),
                    torch.empty((self.batch_size, 0), dtype=torch.long))

        chunk = self.data[self.batch_ind:self.batch_ind + seq_len + 1]
        self.batch_ind += seq_len

        inp = chunk[:-1].t().contiguous()
        targ = chunk[1:].t().contiguous()
        return inp, targ

def load_PTB_char(ptb_path, fname_train, fname_val, fname_test):
    train = read_file(ptb_path + fname_train)
    val = read_file(ptb_path + fname_val)
    test = read_file(ptb_path + fname_test)
    
    chars = sorted(list(set(train + val + test)))
    vocab_size = len(chars)
    print("vocab_size =", vocab_size)

    char_to_id = {ch: i for i, ch in enumerate(chars)}
    id_to_char = {i: ch for i, ch in enumerate(chars)}

    train_ids = [char_to_id[ch] for ch in train]
    val_ids = [char_to_id[ch] for ch in val]
    test_ids = [char_to_id[ch] for ch in test]

    return (torch.tensor(train_ids, dtype=torch.long),
            torch.tensor(val_ids, dtype=torch.long),
            torch.tensor(test_ids, dtype=torch.long),
            char_to_id,
            id_to_char)

class PTB_char:
    """
    Аналог оригинального PTB_char, переписанный на PyTorch.
    Работает с 1D torch.LongTensor (символьная последовательность).
    """
    def __init__(self, data: torch.LongTensor, length: int, batch_size: int, augment: bool = True):
        """
        data: 1D torch.LongTensor (последовательность символов)
        length: длина последовательности (seq_len)
        batch_size: размер батча
        augment: если True — добавляется случайный сдвиг (data augmentation)
        """
        if not torch.is_tensor(data):
            raise TypeError("data must be a torch.Tensor (1D) of long/int")
        if data.dim() != 1:
            raise ValueError("data must be 1D tensor")

        self.data = data
        self.length = int(length)
        self.batch_size = int(batch_size)
        self.augment = bool(augment)

        # Кол-во доступных примеров (отрезков длиной `length`)
        self.num_examples = len(self.data) // self.length
        if self.augment:
            # оставляем 1 длину для случайного смещения
            self.num_examples -= 1

        self.num_batches = int(math.ceil(self.num_examples / float(batch_size)))
        self.max_offset = len(self.data) - self.num_examples * self.length

        # Индексы начала каждого сэмпла
        self.inds = torch.arange(0, self.num_examples) * self.length
        self.inds = [self.inds[i*batch_size : min(self.num_examples, (i+1)*batch_size)] 
                     for i in range(self.num_batches)]

        self.batch_ind = 0

    def new_epoch(self):
        """
        Перемешивает индексы и добавляет случайный сдвиг, если augment=True.
        """
        offset = 0
        if self.augment and self.max_offset > 0:
            offset = int(torch.randint(low=0, high=self.max_offset + 1, size=(1,)).item())

        inds = torch.arange(0, self.num_examples) * self.length + offset
        perm = torch.randperm(self.num_examples)
        inds = inds[perm]

        self.inds = [inds[i*self.batch_size : min(self.num_examples, (i+1)*self.batch_size)]
                     for i in range(self.num_batches)]
        self.batch_ind = 0

    def to_first_batch(self):
        """Сбросить индекс батча к первому."""
        self.batch_ind = 0

    def get_next_batch(self):
        """
        Возвращает (inputs, targets) — оба размера (batch_size, seq_len).
        inputs — символы [:-1], targets — [1:] для предсказания следующего символа.
        """
        if self.batch_ind >= self.num_batches:
            return (torch.empty((self.batch_size, 0), dtype=torch.long),
                    torch.empty((self.batch_size, 0), dtype=torch.long))

        inds = self.inds[self.batch_ind]
        self.batch_ind += 1

        # Формируем батч
        batch = torch.stack([self.data[i:i+self.length] for i in inds])

        x = batch[:, :-1].contiguous()
        y = batch[:, 1:].contiguous()
        return x, y


class PTBLoader:
    def __init__(self, ptb_word, vocab_size, device="cpu"):
        self.ptb = ptb_word
        self.vocab_size = vocab_size
        self.device = device
        self.data = ptb_word.data
        self.num_examples = len(self.data)

    def __iter__(self):
        self.ptb.new_epoch()
        for _ in range(self.ptb.num_batches):
            x_t, y_t = self.ptb.get_next_batch()
            x_safe = torch.remainder(x_t, self.vocab_size)
            x_t = nn.functional.one_hot(x_safe, num_classes=self.vocab_size).float()
            yield x_t, y_t

    def __len__(self):
        return int(self.ptb.num_batches)

    def train_size(self):
        return self.ptb.data.numel()


def generate_vocab_paths(paths, lower=False):
    vocab_freqs = defaultdict(int)
    doc_counts = defaultdict(int)
    if type(paths) == str:
        paths = [paths]
    for path in paths:
            for doc_name in os.listdir(path):
                with open(os.path.join(path, doc_name), encoding="utf-8", errors="ignore") as doc_f:
                    doc = doc_f.read()
                    if lower:
                        doc = doc.lower()
                    doc_seen = set()

                    tokens = [s for s in re.split(r'\W+', doc) 
                                  if s and not s.isspace()]

                    for token in tokens:
                        vocab_freqs[token] += 1
                        if token not in doc_seen:
                            doc_counts[token] += 1
                            doc_seen.add(token)
    return vocab_freqs, doc_counts

def generate_vocab_tab(tab_path, lower=False):
    import csv
    vocab_freqs = defaultdict(int)
    doc_counts = defaultdict(int)
    
    with open(tab_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        # next(reader)  # Пропускаем заголовок, если есть
        for row in reader:
            if not row:
                continue
            label, *text_parts = row
            doc = ' '.join(text_parts)
            if lower:
                doc = doc.lower()
            doc_seen = set()

            tokens = [s for s in re.split(r'\W+', doc) if s and not s.isspace()]
            for token in tokens:
                vocab_freqs[token] += 1
                if token not in doc_seen:
                    doc_counts[token] += 1
                    doc_seen.add(token)
    return vocab_freqs, doc_counts

def generate_vocab(paths=None,
                   max_vocab_size=100*1000,
                   doc_count_threshold=1, 
                   lower=False,
                   eos_symb="</s>",
                   save_path="words.txt",
                   tab_path=None):
    
        if paths is not None:
            vocab_freqs, doc_counts = generate_vocab_paths(paths, lower)
        elif tab_path is not None:
            vocab_freqs, doc_counts = generate_vocab_tab(tab_path, lower)
        else:
            raise ValueError("Either paths or tab must be provided")
        
        vocab_freqs = dict((term, freq) for term, freq in 
                           vocab_freqs.items()
                          if doc_counts[term] > doc_count_threshold)

        # Sort by frequency
        ordered_vocab_freqs = sorted(vocab_freqs.items(),
                                     key=operator.itemgetter(1),
                                     reverse=True)

        ordered_vocab_freqs = ordered_vocab_freqs[:max_vocab_size]
        ordered_vocab_freqs.append((eos_symb, 1))
        
        words = [pair[0] for pair in ordered_vocab_freqs]
        # with open(save_path, "w", encoding="utf-8", errors="ignore") as fout:
        #     fout.write("\n".join(words))
        return ordered_vocab_freqs
        

# def filter_large_embeddings_file(emb_fn, vocab_fn, path_to_save="./", max_vectors_num=52000):
#     words = open(vocab_fn).read().split("\n")
#     words = set(words)
#     FLOAT_SIZE = 4
#     vectors = dict()
    
#     with open(emb_fn, 'rb') as f:
#         c = None

#         # read the header
#         header = b""
#         while c != b"\n":
#             c = f.read(1)
#             header += c

#         num_vectors, vector_len = (int(x) for x in header.split())

#         while len(vectors) < max_vectors_num:
#             word = b""        
#             while True:
#                 c = f.read(1)
#                 if c == b" ":
#                     break
#                 word += c

#             binary_vector = f.read(FLOAT_SIZE * vector_len)
#             if str(word, "utf-8") in words:
#                 vectors[word] = [ struct.unpack_from('f', binary_vector, i)[0] 
#                               for i in range(0, len(binary_vector), FLOAT_SIZE) ]

#             sys.stdout.write("%d, %d%%\r" % (len(vectors), len(vectors) / num_vectors * 100))
#             sys.stdout.flush()
    
#     with open(path_to_save, 'w') as fout:
#         for key in vectors:
#             fout.write(str(key, "utf-8")+";"+str(vectors[key])+"\n")


def process_text_imdb(paths_train, 
                      paths_test, 
                      labels,
                      num_words,
                      maxlen,
                      max_vocab_size=100_000,
                      doc_count_threshold=1,
                      pad_value=0):
    """
    Преобразует тексты IMDB в последовательности индексов слов
    и возвращает torch.Tensor.

    Все последовательности будут выравнены по максимальной длине.
    
    Возвращает:
        x_train, x_test: LongTensor, shape (num_samples, max_seq_len)
        y_train, y_test: LongTensor, shape (num_samples,)
        vocab: list слов, длина = vocab_size
    """
    eos_symb = "</s>"
    lower = False

    # --- generate vocab ---
    vocab_freqs = generate_vocab(paths_train, max_vocab_size,
                                 doc_count_threshold, lower)
    vocab_ids = {w: i for i, (w, _) in enumerate(vocab_freqs)}

    # --- helper ---
    def generate_texts(paths, labels):
        docs, targets = [], []
        for path, label in zip(paths, labels):
            for fname in os.listdir(path):
                with open(os.path.join(path, fname), encoding="utf-8", errors="ignore") as f:
                    text = f.read()
                    if lower:
                        text = text.lower()
                    tokens = [t for t in re.split(r'\W+', text) if t]
                    ids = [vocab_ids[t] for t in tokens if t in vocab_ids]
                    ids.append(vocab_ids[eos_symb])
                    if len(ids) < 2:
                        continue
                    docs.append(ids)
                    targets.append(label)
        return docs, targets

    docs_train, targets_train = generate_texts(paths_train, labels)
    docs_test, targets_test = generate_texts(paths_test, labels)

    # --- pad sequences до максимальной длины ---
    def pad_sequences(docs, pad_value=0):
        max_len = max(len(d) for d in docs)
        padded = torch.full((len(docs), max_len), pad_value, dtype=torch.long)
        for i, d in enumerate(docs):
            padded[i, :len(d)] = torch.tensor(d, dtype=torch.long)
        return padded

    x_train = pad_sequences(docs_train, pad_value)
    x_test = pad_sequences(docs_test, pad_value)
    y_train = torch.tensor(targets_train, dtype=torch.long)
    y_test = torch.tensor(targets_test, dtype=torch.long)
    vocab = [w for w, _ in vocab_freqs]
    
    (X_train, y_train, mask_train), (X_test, y_test, mask_test) = load_matrix(x_train,
                                                                            y_train,
                                                                            x_test, 
                                                                            y_test,
                                                                            num_words,
                                                                            maxlen, 
                                                                            mask=True)

    return (X_train, y_train, mask_train), (X_test, y_test, mask_test)

def load_matrix(x_train, 
                labels_train, 
                x_test, 
                labels_test,
                num_words=None, 
                skip_top=0, 
                maxlen=None,
                seed=113, 
                start_char=1, 
                oov_char=2, 
                index_from=3,
                mask=False):
    """    
    x_train, x_test: list of list of word indices
    labels_train, labels_test: list/array of labels

    Возвращает:
      если mask=False:
        (x_train_tensor, y_train_tensor), (x_test_tensor, y_test_tensor)
      если mask=True:
        (x_train_tensor, y_train_tensor, mask_train), (x_test_tensor, y_test_tensor, mask_test)
    
    Все последовательности выравнены по maxlen.
    """
    rng = np.random.default_rng(seed)

    # --- shuffle ---
    indices = rng.permutation(len(x_train))
    x_train = [x_train[i] for i in indices]
    labels_train = [labels_train[i] for i in indices]

    indices = rng.permutation(len(x_test))
    x_test = [x_test[i] for i in indices]
    labels_test = [labels_test[i] for i in indices]

    # --- объединяем для общего паддинга ---
    xs = x_train + x_test
    labels = labels_train + labels_test

    # --- добавляем start_char / index_from ---
    xs = [[start_char] + [w + index_from for w in x] for x in xs]

    # --- определяем maxlen ---
    if maxlen is None:
        maxlen = max(len(x) for x in xs)

    # --- обрезка и OOV ---
    xs_new = []
    masks = [] if mask else None

    for x in xs:
        x = x[:maxlen]
        if oov_char is not None:
            x = [w if (skip_top <= w < (num_words or max(x)+1)) else oov_char for w in x]
        # паддинг
        x_padded = [0]*(maxlen - len(x)) + x
        xs_new.append(x_padded)
        if mask:
            mask_ = [0]*(maxlen - len(x)) + [1]*len(x)
            masks.append(mask_)

    # --- split обратно на train/test ---
    idx = len(x_train)
    x_train_tensor = torch.tensor(xs_new[:idx], dtype=torch.long)
    y_train_tensor = torch.tensor(labels[:idx], dtype=torch.long)
    x_test_tensor = torch.tensor(xs_new[idx:], dtype=torch.long)
    y_test_tensor = torch.tensor(labels[idx:], dtype=torch.long)

    if mask:
        mask_train = torch.tensor(masks[:idx], dtype=torch.float32)
        mask_test = torch.tensor(masks[idx:], dtype=torch.float32)
        return (x_train_tensor, y_train_tensor, mask_train), (x_test_tensor, y_test_tensor, mask_test)
    else:
        return (x_train_tensor, y_train_tensor), (x_test_tensor, y_test_tensor)


def process_imdb(paths_train, 
                 paths_test, 
                 labels,
                 num_words=None, 
                 maxlen=None,
                 max_vocab_size=100_000,
                 doc_count_threshold=1,
                 val_ratio=0.15,
                 pad_value=0,
                 lower=False,
                 eos_symb="</s>",
                 start_char=1,
                 oov_char=2,
                 index_from=3,
                 skip_top=0):
    """
    Полный pipeline: токенизация → преобразование в индексы → паддинг → создание маски → деление на train/val
    Возвращает:
        (X_train, y_train, mask_train), (X_val, y_val, mask_val), (X_test, y_test, mask_test), vocab
    """

    vocab_freqs = generate_vocab(paths_train, max_vocab_size, doc_count_threshold, lower)
    vocab_ids = {w: i for i, (w, _) in enumerate(vocab_freqs)}

    def texts_to_ids(paths, labels):
        docs, targets = [], []
        for path, label in zip(paths, labels):
            for fname in os.listdir(path):
                with open(os.path.join(path, fname), encoding="utf-8", errors="ignore") as f:
                    text = f.read()
                    if lower:
                        text = text.lower()
                    tokens = [t for t in re.split(r'\W+', text) if t]
                    ids = [vocab_ids[t] for t in tokens if t in vocab_ids]
                    ids.append(vocab_ids[eos_symb])
                    if len(ids) < 2:
                        continue
                    docs.append(ids)
                    targets.append(label)
        return docs, targets

    # --- train/test тексты ---
    docs_train, targets_train = texts_to_ids(paths_train, labels)
    docs_test, targets_test = texts_to_ids(paths_test, labels)

    # --- объединяем для общего паддинга ---
    xs_all = docs_train + docs_test
    labels_all = targets_train + targets_test

    # --- добавляем start_char и index_from ---
    xs_all = [[start_char] + [w + index_from for w in x] for x in xs_all]

    # --- maxlen ---
    if maxlen is None:
        maxlen = max(len(x) for x in xs_all)

    # --- обработка OOV и паддинг ---
    xs_padded, masks = [], []
    for x in xs_all:
        x = x[:maxlen]
        if oov_char is not None:
            x = [w if (skip_top <= w < (num_words or max(x)+1)) else oov_char for w in x]
        padded = [pad_value]*(maxlen - len(x)) + x
        xs_padded.append(padded)
        masks.append([0]*(maxlen - len(x)) + [1]*len(x))

    xs_tensor = torch.tensor(xs_padded, dtype=torch.long)
    labels_tensor = torch.tensor(labels_all, dtype=torch.long)
    masks_tensor = torch.tensor(masks, dtype=torch.float32)

    # --- split обратно на train/test ---
    n_train = len(docs_train)
    x_train_full, X_test = xs_tensor[:n_train], xs_tensor[n_train:]
    y_train_full, y_test = labels_tensor[:n_train], labels_tensor[n_train:]
    mask_train_full, mask_test = masks_tensor[:n_train], masks_tensor[n_train:]

    # --- train/val split ---
    n_val = int(val_ratio * n_train)
    split_idx = n_train - n_val

    X_train, y_train, mask_train = x_train_full[:split_idx], y_train_full[:split_idx], mask_train_full[:split_idx]
    X_val, y_val, mask_val = x_train_full[split_idx:], y_train_full[split_idx:], mask_train_full[split_idx:]

    vocab = [w for w, _ in vocab_freqs]

    return (X_train, y_train, mask_train), (X_val, y_val, mask_val), (X_test, y_test, mask_test), vocab

class Reviews:
    """    
    X, y, mask должны быть torch.Tensor
    mask может быть None
    """
    def __init__(self, X: torch.Tensor, y: torch.Tensor, batch_size: int, mask: torch.Tensor = None, shuffle: bool = False):
        if not isinstance(X, torch.Tensor) or not isinstance(y, torch.Tensor):
            raise TypeError("X and y must be torch.Tensor")
        if mask is not None and not isinstance(mask, torch.Tensor):
            raise TypeError("mask must be torch.Tensor or None")
        
        self.X = X
        self.y = y
        self.mask = mask
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_examples = X.size(0)
        self.num_batches = int(math.ceil(self.num_examples / float(self.batch_size)))
        self.inds = [(i*self.batch_size, min(self.num_examples, (i+1)*self.batch_size))
                     for i in range(self.num_batches)]
        self.batch_ind = 0

    def new_epoch(self):
        """Перемешивает данные, если shuffle=True, и сбрасывает счетчик батча"""
        if self.shuffle:
            perm = torch.randperm(self.num_examples)
            self.X = self.X[perm]
            self.y = self.y[perm]
            if self.mask is not None:
                self.mask = self.mask[perm]
        self.batch_ind = 0

    def to_first_batch(self):
        """Сбрасывает счетчик батча на начало"""
        self.batch_ind = 0

    def get_next_batch(self):
        """Возвращает следующий батч (X_batch, y_batch) или (X_batch, y_batch, mask_batch)"""
        if self.batch_ind >= self.num_batches:
            return None
        
        start, end = self.inds[self.batch_ind]
        self.batch_ind += 1
        
        X_batch = self.X[start:end]
        y_batch = self.y[start:end]
        if self.mask is not None:
            mask_batch = self.mask[start:end]
            return X_batch, y_batch, mask_batch
        else:
            return X_batch, y_batch


class ReviewsLoader:
    """
    DataLoader для Reviews (X, y, mask) с поддержкой:
      - shuffle
      - маски
    """
    def __init__(self, reviews: Reviews):
        """
        reviews: объект класса Reviews
        """
        self.reviews = reviews

    def __iter__(self):
        self.reviews.new_epoch()
        for _ in range(self.reviews.num_batches):
            batch = self.reviews.get_next_batch()
            if batch is None:
                break
            if len(batch) == 3:
                X_batch, y_batch, mask_batch = batch
            else:
                X_batch, y_batch = batch
                mask_batch = None

            if mask_batch is not None:
                yield X_batch, y_batch, mask_batch
            else:
                yield X_batch, y_batch

    def __len__(self):
        return self.reviews.num_batches

    def train_size(self):
        return self.reviews.num_examples