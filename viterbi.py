import sys
from scipy.sparse import csr_matrix, csc_matrix, coo_matrix
import numpy as np


class HMM:
    def __init__(self, initial, transitions, emissions):
        self.symbols = sorted(list(set([sym for tag, sym in emissions.keys()])))
        self.tags = sorted(list(set([t1 for t1, t2 in transitions.keys()] + [t2 for t1, t2 in transitions.keys()])))

        # Make sym and trans 2 idx dictionaries
        self._sym2idx = {sym: idx for idx, sym in enumerate(self.symbols)}
        self._tag2idx = {tag: idx for idx, tag in enumerate(self.tags)}

        self.initial = np.array([initial.get(k, 0) for k in self.tags])

        prob_from_to = [[prob, self.tag2idx(tt[0]), self.tag2idx(tt[1])] for tt, prob in transitions.items()]
        p, t_from, t_to = zip(*prob_from_to)
        self.transitions = csc_matrix((p, (t_from, t_to)), shape=(len(self.tags), len(self.tags))).toarray()

        prob_tag_sym = [[prob, self.tag2idx(st[0]), self.sym2idx(st[1])] for st, prob in emissions.items()]
        p, t, s = zip(*prob_tag_sym)
        #self.emissions = csr_matrix((p, (t, s)), shape=(len(self.tags), len(self.symbols))).T#.toarray()
        self.emissions = csc_matrix((p, (t, s)), shape=(len(self.tags), len(self.symbols))).toarray()

    def sym2idx(self, sym, default=None):
        if sym not in self._sym2idx and default is not None:
            sym = default
        return self._sym2idx.get(sym)

    def tag2idx(self, tag, default=None):
        return self._tag2idx.get(tag)

    def idx2tag(self, idx):
        return self.tags[idx]

    def idx2sym(self, idx):
        return self.symbols[idx]

    def viterbi(self, sentence):
        """Creates a path probability matrix viterbi
        """
        n = self.transitions.shape[0]
        t = len(sentence)
        viterbi = np.zeros((n, t))
        backpointers = np.zeros((n, t))

        #temp = self.initial.multiply(self.emissions[:, self.sym2idx(sentence[0], '<unk>')])
        viterbi[:, 0] = self.initial * self.emissions[:, self.sym2idx(sentence[0], '<unk>')]

        def update_viterbi(word, t):
            """Recursive Step. Updates the possible states for word in test sentence.
            """
            probs = viterbi[:, t-1].reshape(-1, 1) * self.transitions * self.emissions[:, self.sym2idx(word, '<unk>')].reshape(1, -1)
            viterbi[:, t] = np.max(probs, axis=0)
            backpointers[:, t] = np.argmax(probs, axis=0)

        for i, word in enumerate(sentence[1:]):
            update_viterbi(word, i+1)

        return viterbi, backpointers


class HMMParser:
    def __init__(self, input_file_name):
        self.initial = {}
        self.transitions = {}
        self.emissions = {}
        self.states = set()
        self.header = {}
        self.symbols = set()

        self.parse_hmm(input_file_name)

    def parse_hmm(self, input_hmm):
        """Parse HMM and store in dictionaries: inital probabilities, transitions, and emissions.
        """
        in_init = False
        in_transition = False
        in_emission = False
        with open(input_hmm) as f:
            for line in f:
                line = line.strip()
                try:
                    k, v = line.split('=')
                    self.header[k] = int(v)
                except ValueError:
                    pass
                # Deal with those \ slashed lines
                if line.startswith('\\'):
                    if 'init' in line:
                        in_init = True
                    if 'transition' in line:
                        in_transition = True
                    if 'emission' in line:
                        in_emission = True
                    continue
                # Count line numbers for each category
                if in_init:
                    if not line:
                        in_init = False
                        continue
                    state, probability = line.split()
                    self.initial[state] = float(probability)

                # Most lines: Add unique State into dictionary
                if in_transition:
                    if not line:
                        in_transition = False
                        continue
                    from_state, to_state, probability, *_ = line.split()
                    self.transitions[(from_state, to_state)] = float(probability)
                    self.states.add(from_state)
                    self.states.add(to_state)

                if in_emission:
                    if not line:
                        in_emission = False
                        continue
                    from_state, symbol, probability, *_ = line.split()
                    self.emissions[(from_state, symbol)] = float(probability)
                    self.symbols.add(symbol)
                    self.states.add(from_state)


def parse_test(test_file):
    """Parse test_file with line separated output symbols.
    """
    with open(test_file) as f:
        for line in f:
            line = line.strip().split()
            yield ['<s>'] + line + ['</s>']


def get_best_path(hmm, sentence):
    viterbi, backpointers = hmm.viterbi(sentence)
    state_sequence = []
    state_probs = []
    curr_state = int(np.argmax(viterbi[:, -1]))
    for t in range(viterbi.shape[1] - 1, -1, -1):
        try:
            state_sequence.append(curr_state)
            state_probs.append(viterbi[curr_state, t])
            curr_state = int(backpointers[curr_state, t])
        except IndexError:
            print(curr_state, t)
            raise
    state_sequence = state_sequence[::-1]
    state_probs = state_probs[::-1]
    tag_sequence = list(map(hmm.idx2tag, state_sequence))
    lg_prob = np.log10(state_probs[-1])
    return tag_sequence, lg_prob


def main():
    input_hmm = sys.argv[1]
    hmm_parser = HMMParser(input_hmm)
    hmm = HMM(hmm_parser.initial, hmm_parser.transitions, hmm_parser.emissions)
    test_file = sys.argv[2]
    for sentence in parse_test(test_file):
        tag_sequence, lg_prob = get_best_path(hmm, sentence)
        print(" ".join(sentence[1:-1]), "=>", " ".join(tag_sequence[:-1]), lg_prob)


if __name__ == "__main__":
    main()
