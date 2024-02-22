from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pickle

path = f"data/answer_synthetic_data_from_M_HalDetect.bin"
with open(path, "rb") as f:
    responses = pickle.load(f)

for idx, response in responses.items():
    qingli = 3



# # Load the tokenizer and model
# tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")
# model = AutoModelForCausalLM.from_pretrained("huggyllama/llama-7b")
#
# # Example input text
# text = "The quick brown fox jumps over the lazy dog."
#
# # Tokenize input
# tokens = tokenizer(text, return_tensors="pt")
# input_ids = tokens.input_ids
#
# # Perform a forward pass to get logits
# with torch.no_grad():
#     outputs = model(**tokens)
#     logits = outputs.logits
#
# # Shift the logits and input_ids by one position so that we align the logits with their respective tokens
# shifted_logits = logits[:, :-1, :]
# shifted_input_ids = input_ids[:, 1:]
#
# # Convert logits to probabilities
# log_probs = torch.nn.functional.log_softmax(shifted_logits, dim=-1)
#
# # Gather the log probabilities for the actual next tokens
# gathered_log_probs = torch.gather(log_probs, 2, shifted_input_ids.unsqueeze(-1)).squeeze(-1)
#
# # Calculate the average log probability across the sequence
# average_log_prob = gathered_log_probs.sum(1) / (input_ids.size(1) - 1)
#
# # Calculate perplexity
# perplexity = torch.exp(-average_log_prob)
#
# # Print perplexity
# print("Perplexity:", perplexity.tolist())
