import os
import torch
import collections

class LabelConverter(object):
    """Convert between str and label.

    NOTE:
        Insert `blank` to the alphabet for CTC.

    Args:
        alphabet (str): set of the possible characters.
        ignore_case (bool, default=True): whether or not to
            ignore all of the case.
    """

    def __init__(self, alphabet, ignore_case=True):
        self._ignore_case = ignore_case
        if self._ignore_case:
            alphabet = alphabet.lower()
        self.alphabet = alphabet + '-'  # for `-1` index

        self.dict = {}
        for i, char in enumerate(alphabet):
            # NOTE: 0 is reserved for 'blank' required by wrap_ctc
            self.dict[char] = i + 1

    def encode(self, labels):
        """Support batch or single str.

        Args:
            labels (str or list of str): labels to convert.

        Returns:
            torch.IntTensor [length_0 + length_1 + ...
                length_{n - 1}]: encoded labels.
            torch.IntTensor [n]: length of each labels.
        """
        if isinstance(labels, str):
            labels = [
                self.dict[char.lower() if self._ignore_case else char]
                for char in labels
            ]
            length = [len(labels)]
        elif isinstance(labels, collections.Iterable):
            length = [len(s) for s in labels]
            labels = ''.join(labels)
            labels, _ = self.encode(labels)
        return (torch.IntTensor(labels), torch.IntTensor(length))

    def decode(self, probs, length, raw=False):
        """Decode encoded labels back into strs.

        Args:
            torch.IntTensor [length_0 + length_1 + ...
                length_{n - 1}]: encoded labels.
            torch.IntTensor [n]: length of each labels.

        Raises:
            AssertionError: when the labels and its length does not match.

        Returns:
            labels (str or list of str): labels to convert.
        """
        if length.numel() == 1:
            length = length[0]
            assert probs.numel() == length
            if raw:
                return ''.join([self.alphabet[i - 1] for i in probs])
            else:
                char_list = []
                for i in range(length):
                    if (probs[i] != 0 and (not (i > 0 and probs[i - 1] == probs[i]))):
                        char_list.append(self.alphabet[probs[i] - 1])
                return ''.join(char_list)
        else:
            # batch mode
            assert probs.numel() == length.sum()
            labels = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                labels.append(self.decode(probs[index:index + l], torch.IntTensor([l]), raw=raw))
                index += l
            return labels

    def predict(self, probs):
        seq_len, batch_size = probs.shape[:2]
        lengths = torch.IntTensor(batch_size).fill_(seq_len)
        _, probs = probs.max(2)
        probs = probs.transpose(1, 0).contiguous().reshape(-1)
        preds = self.decode(probs, lengths)
        return preds
