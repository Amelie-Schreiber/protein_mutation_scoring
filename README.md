# Protein Mutation Scoring
Some scoring functions for predicting the effects of mutations on protein sequences using ESM-2. 

## Usage
Below are examples on how to run different scoring methods on the `mutations.csv` file. Valid models are:

- `facebook/esm2_t6_8M_UR50D`, 
- `facebook/esm2_t12_35M_UR50D`,
- `facebook/esm2_t30_150M_UR50D`,
- `facebook/esm2_t33_650M_UR50D`.

You can change the protein sequence to a sequence of your choice, just be sure to update the `mutations.csv` file to mutations that make sense for your protein. The script zero-indexes the residues by default. So the if you mutate the first residue, this is the $0^{th}$ residue. Adjust the `output.csv` to whatever name you want for the file you want the output scores to be in. 
Example usage (masked marginals):
```
python3 scoring_esm2.py \
  --model "facebook/esm2_t12_35M_UR50D" \
  --sequence "MKTIIALSYIFCLVFA" \
  --dms-input mutations.csv \
  --dms-output output.csv \
  --mutation-col mutant \
  --scoring-strategy masked-marginals 
```
Example usage (wild-type marginals):
```
python3 scoring_esm2.py \
  --model "facebook/esm2_t12_35M_UR50D" \
  --sequence "MKTIIALSYIFCLVFA" \
  --dms-input mutations.csv \
  --dms-output output.csv \
  --mutation-col mutant \
  --scoring-strategy wt-marginals 
```
Example usage (pseudo-perplexity):
```
python3 scoring_esm2.py \
  --model "facebook/esm2_t12_35M_UR50D" \
  --sequence "MKTIIALSYIFCLVFA" \
  --dms-input mutations.csv \
  --dms-output output.csv \
  --mutation-col mutant \
  --scoring-strategy pseudo-ppl 
```

