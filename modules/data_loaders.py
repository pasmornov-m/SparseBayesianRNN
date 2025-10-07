import torch
import torch.nn as nn
import math
from typing import Tuple


def read_words(ptb_path: str, fname: str):
        with open(ptb_path + fname, "r") as fin:
            raw = fin.read()[::2]
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
    VOCAB_SIZE = len(vocab)
    print("VOCAB_SIZE =", VOCAB_SIZE)

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


class PTBWordLoader:
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
            x_onehot = nn.functional.one_hot(x_t, num_classes=self.vocab_size).float()
            yield x_onehot, y_t

    def __len__(self):
        return int(self.ptb.num_batches)

    def train_size(self):
        return self.ptb.data.numel()