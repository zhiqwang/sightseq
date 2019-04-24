import torch
from collections.abc import Iterable


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
            # NOTE: 0 is reserved for 'blank' required by CTCLoss
            self.dict[char] = i + 1

    def encode(self, labels):
        """Support batch or single str.

        Args:
            labels (str or list of str): labels to convert.

        Returns:
            torch.IntTensor [length_0 + length_1 + ... length_{n-1}]: encoded labels.
            torch.IntTensor [n]: length of each labels.
        """
        if isinstance(labels, str):
            labels = [self.dict[char.lower() if self._ignore_case else char] for char in labels]
            length = [len(labels)]
        elif isinstance(labels, Iterable):
            length = [len(s) for s in labels]
            labels = ''.join(labels)
            labels, _ = self.encode(labels)
        return (torch.IntTensor(labels), torch.IntTensor(length))

    def decode(self, probs, length, raw=False, strings=True):
        """Decode encoded labels back into strings.

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
                if strings:
                    return u''.join([self.alphabet[i - 1] for i in probs]).encode('utf-8')
                return probs.tolist()
            else:
                probs_non_blank = []
                for i in range(length):
                    # removing repeated characters and blank.
                    if (probs[i] != 0 and (not (i > 0 and probs[i - 1] == probs[i]))):
                        if strings:
                            probs_non_blank.append(self.alphabet[probs[i] - 1])
                        else:
                            probs_non_blank.append(probs[i].item())
                if strings:
                    return u''.join(probs_non_blank).encode('utf-8')
                return probs_non_blank
        else:
            # batch mode
            assert probs.numel() == length.sum()
            labels = []
            index = 0
            for i in range(length.numel()):
                idx_end = length[i]
                labels.append(self.decode(probs[index:index + idx_end],
                              torch.IntTensor([idx_end]), raw=raw, strings=strings))
                index += idx_end
            return labels

    def best_path_decode(self, probs, raw=False, strings=True):
        lengths = torch.full((probs.shape[1],), probs.shape[0], dtype=torch.int32)
        _, probs = probs.max(2)
        probs = probs.transpose(1, 0).contiguous().reshape(-1)
        preds = self.decode(probs, lengths, raw=raw, strings=strings)
        return preds
