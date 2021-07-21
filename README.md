# viterbi_HMM

Files:
viterbi.sh; viterbi.py
conv_format.sh; conv_format.py

Output files:
  sys1; sys1_res; sys1_res.acc
  sys2; sys2_res; sys2_res.acc
  sys3; sys3_res; sys3_res.acc
  sys4; sys4_res; sys4_res.acc
  sys5; sys5_res; sys5_res.acc

cat sys1 | bash conv_format.sh sys1_res

## Q1: viterbi.sh
In our HW8 system, we implement the Viterbi algorithm for a Hidden Markov Model (HMM) and calculate the best path and its probability.
We represent the most probable path by taking the max path through all possible State sequences, for an observation of length(t).
We also store Backpointers to produce the most likely State sequence. We return both the max path and its probability.

## Q2: conv_format.sh
To use viterbi.sh for trigram POS tagging, we convert our Q1 output_file to word/tag pairs.

Table 1: Tagging accuracy

  HMM model   tagging accuracy

    hmm1       81.9518170681911

    hmm2       87.1376071866068

    hmm3       88.3217639853001

    hmm4       87.9542670477746

    hmm5       88.1584320130666



Issue:
reshaping() not implemented in scipy.sparse on patas.
Changed to np.arrrays
