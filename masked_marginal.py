import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from typing import List, Tuple

def masked_marginal_scoring(sequence: str, mutations: List[str], model, tokenizer) -> List[Tuple[str, float]]:
    """
    Calculates the masked marginal scores for a list of mutations in a given protein sequence.

    :param sequence: The original protein sequence.
    :param mutations: A list of mutations (e.g., ["A1B", "C3D"]).
    :param model: The loaded ESM model.
    :param tokenizer: The tokenizer corresponding to the model.

    :return: A list of tuples, where each tuple contains a mutation and its masked marginal score.
    """

    # Move model to the appropriate device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Prepare the base sequence
    data = [("protein1", sequence)]
    batch_tokens = tokenizer.batch_encode_plus(data, return_tensors="pt", padding=True)["input_ids"].to(device)

    scores = []
    for mutation in mutations:
        wt, idx_str, mt = mutation[0], mutation[1:-1], mutation[-1]
        idx = int(idx_str) - 1  # Adjust for 0-based indexing

        # Ensure the sequence matches the expected wild type at the specified index
        assert sequence[idx] == wt, f"The listed wildtype {wt} does not match the sequence at position {idx_str}"

        # Perform masking
        batch_tokens_masked = batch_tokens.clone()
        batch_tokens_masked[0, 1 + idx] = tokenizer.mask_token_id  # 1 added for BOS token

        with torch.no_grad():
            # Apply log_softmax to convert logits to log probabilities
            token_probs = torch.log_softmax(model(batch_tokens_masked).logits, dim=-1)

        mt_encoded = tokenizer.encode(mt, add_special_tokens=False)[0]
        score = token_probs[0, 1 + idx, mt_encoded].item() - token_probs[0, 1 + idx, tokenizer.encode(wt, add_special_tokens=False)[0]].item()
        scores.append((mutation, score))
        mm_score = sum(score for _, score in scores)

    return scores, mm_score

# Example usage
# model_name = "facebook/esm2_t12_35M_UR50D"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForMaskedLM.from_pretrained(model_name)

# sequence = "MKTIIALSYIFCLVFA"
# mutations = ["T3B", "I4A", "A6M"]  # Example mutations
# scores, mm_score = masked_marginal_scoring(sequence, mutations, model, tokenizer)

# for mutation, score in scores:
#     print(f"Mutation: {mutation}, Score: {score}")
# print(f"Masked Marginal Score: {mm_score}")
