import torch
import torch.nn as nn
import math
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
            # x_t = nn.functional.one_hot(x_t, num_classes=self.vocab_size).float()
            # x_safe = x_t % self.vocab_size
            x_safe = torch.remainder(x_t, self.vocab_size)
            x_t = nn.functional.one_hot(x_safe, num_classes=self.vocab_size).float()
            yield x_t, y_t

    def __len__(self):
        return int(self.ptb.num_batches)

    def train_size(self):
        return self.ptb.data.numel()