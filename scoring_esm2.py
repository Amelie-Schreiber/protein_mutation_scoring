import argparse
import pathlib
import string
import torch
from esm import Alphabet, pretrained, MSATransformer
import pandas as pd
from tqdm import tqdm
from Bio import SeqIO
import itertools

def remove_insertions(sequence: str) -> str:
    deletekeys = dict.fromkeys(string.ascii_lowercase)
    deletekeys["."] = None
    deletekeys["*"] = None
    translation = str.maketrans(deletekeys)
    return sequence.translate(translation)

def create_parser():
    parser = argparse.ArgumentParser(description="Label a deep mutational scan with predictions from an ensemble of ESM-1v models.")
    parser.add_argument("--model-location", type=str, help="PyTorch model file OR name of pretrained model to download", nargs="+")
    parser.add_argument("--sequence", type=str, help="Base sequence to which mutations were applied")
    parser.add_argument("--dms-input", type=pathlib.Path, help="CSV file containing the deep mutational scan")
    parser.add_argument("--mutation-col", type=str, default="mutant", help="column in the deep mutational scan labeling the mutation as 'AiB'")
    parser.add_argument("--dms-output", type=pathlib.Path, help="Output file containing the deep mutational scan along with predictions")
    parser.add_argument("--offset-idx", type=int, default=0, help="Offset of the mutation positions in `--mutation-col`")
    parser.add_argument("--scoring-strategy", type=str, default="wt-marginals", choices=["wt-marginals", "pseudo-ppl", "masked-marginals"], help="")
    parser.add_argument("--msa-path", type=pathlib.Path, help="path to MSA in a3m format (required for MSA Transformer)")
    parser.add_argument("--msa-samples", type=int, default=400, help="number of sequences to select from the start of the MSA")
    parser.add_argument("--nogpu", action="store_true", help="Do not use GPU even if available")
    return parser

def label_row(row, sequence, token_probs, alphabet, offset_idx):
    wt, idx, mt = row[0], int(row[1:-1]) - offset_idx, row[-1]
    assert sequence[idx] == wt, "The listed wildtype does not match the provided sequence"
    wt_encoded, mt_encoded = alphabet.get_idx(wt), alphabet.get_idx(mt)
    score = token_probs[0, 1 + idx, mt_encoded] - token_probs[0, 1 + idx, wt_encoded]
    return score.item()

def compute_pppl(row, sequence, model, alphabet, offset_idx, device):
    wt, idx, mt = row[0], int(row[1:-1]) - offset_idx, row[-1]
    assert sequence[idx] == wt, "The listed wildtype does not match the provided sequence"

    # modify the sequence
    sequence = sequence[:idx] + mt + sequence[(idx + 1):]

    # encode the sequence
    data = [("protein1", sequence)]

    batch_converter = alphabet.get_batch_converter()
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(device)  # Move batch_tokens to the same device

    log_probs = []
    for i in range(1, len(sequence) - 1):
        batch_tokens_masked = batch_tokens.clone()
        batch_tokens_masked[0, i] = alphabet.mask_idx
        with torch.no_grad():
            token_probs = torch.log_softmax(model(batch_tokens_masked)["logits"], dim=-1)
        log_probs.append(token_probs[0, i, alphabet.get_idx(sequence[i])].item())
    return sum(log_probs)


def main(args):
    df = pd.read_csv(args.dms_input)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.nogpu else "cpu")

    for model_location in args.model_location:
        model, alphabet = pretrained.load_model_and_alphabet(model_location)
        model = model.to(device)
        model.eval()
        batch_converter = alphabet.get_batch_converter()

        if isinstance(model, MSATransformer):
            data = [read_msa(args.msa_path, args.msa_samples)]
            assert args.scoring_strategy == "masked-marginals", "MSA Transformer only supports masked marginal strategy"
            batch_labels, batch_strs, batch_tokens = batch_converter(data)
            batch_tokens = batch_tokens.to(device)

            all_token_probs = []
            for i in tqdm(range(batch_tokens.size(2))):
                batch_tokens_masked = batch_tokens.clone()
                batch_tokens_masked[0, 0, i] = alphabet.mask_idx
                with torch.no_grad():
                    token_probs = torch.log_softmax(model(batch_tokens_masked)["logits"], dim=-1)
                all_token_probs.append(token_probs[:, 0, i])
            token_probs = torch.cat(all_token_probs, dim=0).unsqueeze(0)
            df[model_location] = df.apply(
                lambda row: label_row(row[args.mutation_col], args.sequence, token_probs, alphabet, args.offset_idx),
                axis=1,
            )
        else:
            data = [("protein1", args.sequence)]
            batch_labels, batch_strs, batch_tokens = batch_converter(data)
            batch_tokens = batch_tokens.to(device)

            if args.scoring_strategy == "wt-marginals":
                with torch.no_grad():
                    token_probs = torch.log_softmax(model(batch_tokens)["logits"], dim=-1)
                df[model_location] = df.apply(
                    lambda row: label_row(row[args.mutation_col], args.sequence, token_probs, alphabet, args.offset_idx),
                    axis=1,
                )
            elif args.scoring_strategy == "masked-marginals":
                all_token_probs = []
                for i in tqdm(range(batch_tokens.size(1))):
                    batch_tokens_masked = batch_tokens.clone()
                    batch_tokens_masked[0, i] = alphabet.mask_idx
                    with torch.no_grad():
                        token_probs = torch.log_softmax(model(batch_tokens_masked)["logits"], dim=-1)
                    all_token_probs.append(token_probs[:, i])
                token_probs = torch.cat(all_token_probs, dim=0).unsqueeze(0)
                df[model_location] = df.apply(
                    lambda row: label_row(row[args.mutation_col], args.sequence, token_probs, alphabet, args.offset_idx),
                    axis=1,
                )
            elif args.scoring_strategy == "pseudo-ppl":
                tqdm.pandas()
                df[model_location] = df.progress_apply(
                    lambda row: compute_pppl(row[args.mutation_col], args.sequence, model, alphabet, args.offset_idx, device),
                    axis=1,
                )

    df.to_csv(args.dms_output)

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
