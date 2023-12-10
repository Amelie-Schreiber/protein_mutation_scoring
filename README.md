# Protein Mutation Scoring
Some scoring functions for predicting the effects of mutations on protein sequences using ESM-2. 

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

