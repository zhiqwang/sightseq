import numpy as np


class BeamEntry:
    '''information about one single beam at specific time-step'''
    def __init__(self):
        self.pr_total = 0  # blank and non-blank
        self.pr_non_blank = 0  # non-blank
        self.pr_blank = 0  # blank
        self.pr_text = 1  # LM score
        self.lm_applied = False  # flag if LM was already applied to this beam
        self.labeling = ()  # beam-labeling


class BeamState:
    '''information about the beams at specific time-step'''
    def __init__(self):
        self.entries = {}

    def norm(self):
        '''length-normalise LM score'''
        for (k, _) in self.entries.items():
            labeling_len = len(self.entries[k].labeling)
            self.entries[k].pr_text = self.entries[k].pr_text ** (1.0 / (labeling_len if labeling_len else 1.0))

    def sort(self):
        '''return beam-labelings, sorted by probability'''
        beams = [v for (_, v) in self.entries.items()]
        sorted_beams = sorted(beams, reverse=True, key=lambda x: x.pr_total * x.pr_text)
        return [x.labeling for x in sorted_beams]


def ctc_beam_search(mat, classes, lm):
    '''beam search as described by the paper
    of Hwang et al. and the paper of Graves et al.
    '''
    blankIdx = len(classes)
    max_t, max_c = mat.shape
    beam_width = 25

    # initialise beam state
    last = BeamState()
    labeling = ()
    last.entries[labeling] = BeamEntry()
    last.entries[labeling].pr_blank = 1
    last.entries[labeling].pr_total = 1

    # go over all time-steps
    for t in range(max_t):
        curr = BeamState()

        # get beam-labelings of best beams
        bestLabelings = last.sort()[0:beam_width]

        # go over best beams
        for labeling in bestLabelings:

            # probability of paths ending with a non-blank
            pr_non_blank = 0
            # in case of non-empty beam
            if labeling:
                # probability of paths with repeated last char at the end
                pr_non_blank = last.entries[labeling].pr_non_blank * mat[t, labeling[-1]]

            # probability of paths ending with a blank
            pr_blank = (last.entries[labeling].pr_total) * mat[t, blankIdx]

            # add beam at current time-step if needed
            add_beam(curr, labeling)

            # fill in data
            curr.entries[labeling].labeling = labeling
            curr.entries[labeling].pr_non_blank += pr_non_blank
            curr.entries[labeling].pr_blank += pr_blank
            curr.entries[labeling].pr_total += pr_blank + pr_non_blank
            # beam-labeling not changed, therefore also LM score unchanged from
            curr.entries[labeling].pr_text = last.entries[labeling].pr_text
            # LM already applied at previous time-step for this beam-labeling
            curr.entries[labeling].lm_applied = True

            # extend current beam-labeling
            for c in range(max_c - 1):
                # add new char to current beam-labeling
                new_labeling = labeling + (c,)

                # if new labeling contains duplicate char at the end,
                # only consider paths ending with a blank
                if labeling and labeling[-1] == c:
                    pr_non_blank = mat[t, c] * last.entries[labeling].pr_blank
                else:
                    pr_non_blank = mat[t, c] * last.entries[labeling].pr_total

                # add beam at current time-step if needed
                add_beam(curr, new_labeling)

                # fill in data
                curr.entries[new_labeling].labeling = new_labeling
                curr.entries[new_labeling].pr_non_blank += pr_non_blank
                curr.entries[new_labeling].pr_total += pr_non_blank

                # apply LM
                apply_lm(curr.entries[labeling], curr.entries[new_labeling], classes, lm)

        # set new beam state
        last = curr

    # normalise LM scores according to beam-labeling-length
    last.norm()

    # sort by probability
    best_labeling = last.sort()[0]  # get most probable labeling

    # map labels to chars
    res = ''
    for l in best_labeling:
        res += classes[l]

    return res


def apply_lm(parent_beam, child_beam, classes, lm):
    '''calculate LM score of child beam by taking score from parent
    beam and bigram probability of last two chars
    '''
    if lm and not child_beam.lm_applied:
        # first char
        c1 = classes[parent_beam.labeling[-1] if parent_beam.labeling else classes.index(' ')]
        # second char
        c2 = classes[child_beam.labeling[-1]]
        # influence of language model
        lm_factor = 0.01
        # probability of seeing first and second char next to each other
        bigram_prob = lm.getCharBigram(c1, c2) ** lm_factor
        # probability of char sequence
        child_beam.pr_text = parent_beam.pr_text * bigram_prob
        # only apply LM once per beam entry
        child_beam.lm_applied = True


def add_beam(beam_state, labeling):
    '''add beam if it does not yet exist'''
    if labeling not in beam_state.entries:
        beam_state.entries[labeling] = BeamEntry()


def test_beam_search():
    '''test decoder'''
    classes = 'ab'
    mat = np.array([[0.1, 0, 0.9], [0.4, 0, 0.6]])
    print('Test beam search')
    expected = 'a'
    actual = ctc_beam_search(mat, classes, None)
    print('Expected: "' + expected + '"')
    print('Actual: "' + actual + '"')
    print('OK' if expected == actual else 'ERROR')


if __name__ == '__main__':
    test_beam_search()
