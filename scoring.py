import argparse
import pathlib
import string
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
import pandas as pd
from tqdm import tqdm
from Bio import SeqIO
import itertools
from typing import List, Tuple

# Removes insertions from a sequence, needed for aligned sequences in MSA processing
def remove_insertions(sequence: str) -> str:
    # Delete lowercase characters and insertion characters ('.', '*') from the string
    deletekeys = dict.fromkeys(string.ascii_lowercase)
    deletekeys["."] = None
    deletekeys["*"] = None
    translation = str.maketrans(deletekeys)
    return sequence.translate(translation)

# Create and configure the argument parser for command line interface
def create_parser():
    parser = argparse.ArgumentParser(
        description="Label a deep mutational scan with predictions from various ESM models."
    )
    # Define the arguments the script can accept
    parser.add_argument("--model", type=str, default="facebook/esm2_t6_8M_UR50D",
                        choices=["facebook/esm2_t6_8M_UR50D", "facebook/esm2_t12_35M_UR50D", 
                                 "facebook/esm2_t30_150M_UR50D", "facebook/esm2_t33_650M_UR50D"],
                        help="ESM model to be used.")
    parser.add_argument("--sequence", type=str, help="Base sequence to which mutations were applied")
    parser.add_argument("--dms-input", type=pathlib.Path, help="CSV file containing the deep mutational scan")
    parser.add_argument("--mutation-col", type=str, default="mutant", help="Column in the deep mutational scan labeling the mutation")
    parser.add_argument("--dms-output", type=pathlib.Path, help="Output file containing the deep mutational scan along with predictions")
    parser.add_argument("--offset-idx", type=int, default=0, help="Offset of the mutation positions")
    parser.add_argument("--scoring-strategy", type=str, default="wt-marginals", choices=["wt-marginals", "masked-marginals", "pseudo-ppl"], help="Scoring strategy to use")
    parser.add_argument("--nogpu", action="store_true", help="Do not use GPU even if available")
    return parser

# Function to label a row in the DataFrame based on the scoring of mutations
def label_row(row, sequence, token_probs, tokenizer, offset_idx):
    # Extract wild type, index, and mutated type from the row
    wt, idx, mt = row[0], int(row[1:-1]) - offset_idx, row[-1]
    assert sequence[idx] == wt, "The listed wildtype does not match the provided sequence"
    # Encode the wild type and mutated type
    wt_encoded, mt_encoded = tokenizer.encode(wt, add_special_tokens=False)[0], tokenizer.encode(mt, add_special_tokens=False)[0]
    # Calculate the score as the difference in log probabilities
    score = token_probs[0, 1 + idx, mt_encoded] - token_probs[0, 1 + idx, wt_encoded]
    return score.item()

# Function to compute pseudo-perplexity for a row, used in language model evaluation
def compute_pppl(row, sequence, model, tokenizer, offset_idx):
    wt, idx, mt = row[0], int(row[1:-1]) - offset_idx, row[-1]
    assert sequence[idx] == wt, "The listed wildtype does not match the provided sequence"
    # Modify the sequence with the mutation
    sequence = sequence[:idx] + mt + sequence[(idx + 1) :]
    # Tokenize the modified sequence
    data = [("protein1", sequence)]
    batch_tokens = tokenizer.batch_encode_plus(data, return_tensors="pt", padding=True)["input_ids"]
    # Calculate log probabilities for each position
    log_probs = []
    for i in range(1, len(sequence) - 1):
        batch_tokens_masked = batch_tokens.clone()
        batch_tokens_masked[0, i] = tokenizer.mask_token_id
        with torch.no_grad():
            token_probs = torch.log_softmax(model(batch_tokens_masked.to(device)).logits, dim=-1)
        log_probs.append(token_probs[0, i, batch_tokens[0, i]].item())
    return sum(log_probs)

# Main function to orchestrate mutation scoring process
def main(args):
    # Load deep mutational scan data from a CSV file
    df = pd.read_csv(args.dms_input)

    # Determine to use GPU or CPU
    device = torch.device("cuda" if torch.cuda.is_available() and not args.nogpu else "cpu")

    # Load the chosen ESM model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForMaskedLM.from_pretrained(args.model).to(device)
    model.eval()

    # Preprocess and encode the base sequence
    data = [("protein1", args.sequence)]
    batch_tokens = tokenizer.batch_encode_plus(data, return_tensors="pt", padding=True)["input_ids"].to(device)

    # Apply selected scoring strategy
    if args.scoring_strategy == "wt-marginals":
        with torch.no_grad():
            token_probs = torch.log_softmax(model(batch_tokens).logits, dim=-1)
        df[args.model] = df.apply(
            lambda row: label_row(row[args.mutation_col], args.sequence, token_probs, tokenizer, args.offset_idx),
            axis=1
        )
    elif args.scoring_strategy == "masked-marginals":
        all_token_probs = []
        for i in tqdm(range(batch_tokens.size(1))):
            batch_tokens_masked = batch_tokens.clone()
            batch_tokens_masked[0, i] = tokenizer.mask_token_id
            with torch.no_grad():
                token_probs = torch.log_softmax(model(batch_tokens_masked).logits, dim=-1)
            all_token_probs.append(token_probs[:, i])
        token_probs = torch.cat(all_token_probs, dim=0).unsqueeze(0)
        df[args.model] = df.apply(
            lambda row: label_row(row[args.mutation_col], args.sequence, token_probs, tokenizer, args.offset_idx),
            axis=1
        )
    elif args.scoring_strategy == "pseudo-ppl":
        tqdm.pandas()
        df[args.model] = df.progress_apply(
            lambda row: compute_pppl(row[args.mutation_col], args.sequence, model, tokenizer, args.offset_idx),
            axis=1
        )

    # Save the scored mutations to a CSV file
    df.to_csv(args.dms_output)

# Script entry point for command-line interaction
if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)

# Example usage:
# python3 scoring.py --model "facebook/esm2_t12_35M_UR50D" --sequence "MKTIIALSYIFCLVFA" --dms-input mutations.csv --dms-output output.csv --mutation-col mutant --scoring-strategy masked-marginals
