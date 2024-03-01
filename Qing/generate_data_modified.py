import numpy as np
import json
import re

def split_sentences(text):
    # Define a regular expression pattern for matching sentence endings
    # and keep the punctuations after each sentence.
    sentence_endings = r'(?<=[.!?])\s*(?=\b|\s)'

    # Use the re.split() function to split the text based on the sentence endings pattern
    sentences = re.split(sentence_endings, text)

    # Iterate through the sentences to preserve trailing punctuation and whitespace
    for i in range(len(sentences) - 1):
        # Append the next non-space character after the punctuation to the current sentence
        next_char_index = re.search(r'\S', sentences[i + 1]).start()
        sentences[i] += text[len(sentences[i]):next_char_index]

    return sentences


def split_sentences_with_index(text):
    # Define a regular expression pattern for matching sentence endings
    # and keep the punctuations after each sentence.
    sentence_endings = r'(?<=[.!?])\s*(?=\b|\s)'

    # Use the re.finditer() function to find all occurrences of sentence endings
    matches = re.finditer(sentence_endings, text)

    # Store the indices of the first character of each sentence
    indices = [0]  # Start index of the first sentence
    for match in matches:
        indices.append(match.end())

    # Add the end index of the paragraph
    indices.append(len(text))


    # Iterate through the indices to extract each sentence
    sentences = []
    for i in range(len(indices) - 1):
        start_index = indices[i]
        end_index = indices[i + 1]
        sentence = text[start_index:end_index].strip()
        sentences.append(sentence)
        # sentences_with_index.append((start_index, sentence))  # Store index and sentence

    return sentences, indices

def check_consistency(response, sentences):
    no_empty_response = "".join(response.split(" "))
    concat_sents = "".join(sentences)
    no_empty_sents = "".join(concat_sents.split(" "))
    return (no_empty_sents == no_empty_response)



if __name__ == "__main__":

    split = "val"
    new_file_name = f"data/synthetic_{split}_data_from_M_HalDetect_modified.json"
    new_file = open(new_file_name, "w")
    images = []
    ori_file_name = f'data/{split}_raw.json'
    idx = 0
    total_cases = 0
    total_inner_fails, total_outer_fails = 0, 0
    total_successes = 0
    with open(ori_file_name, 'r') as f:
        for file in f.readlines():
            dic = json.loads(file)
            for line in dic:
                question = line['question'].strip().replace("<image>", "").replace("\n", "")
                response = line['response']
                image = line['image'].strip()
                # question_id = line['image'].split(".")[0]
                question_id = idx
                sentences = []
                labels = []

                sentences, indices = split_sentences_with_index(response)
                sent_idx = 0
                labels = [[] for sent in sentences]
                print("consistency:", check_consistency(response, sentences))

                drop = False
                old_sentences = []
                old_labels = []
                total_cases += 1
                for ele in line['annotations']:
                    sub_sent = ele['text'].strip()
                    label = ele['label'].strip()

                    old_sentences.append(sub_sent)
                    old_labels.append(label)

                    while True:
                        try:
                            if ele['start'] >= indices[sent_idx] and ele['end'] <= indices[sent_idx+1]:
                                labels[sent_idx].append(label)
                                break
                            else:
                                sent_idx += 1
                        except:
                            print("failed cases")
                            total_inner_fails += 1
                            drop = True
                            break

                if drop:
                    total_outer_fails += 1
                    continue

                total_successes += 1
                new_labels = ["INACCURATE" if "INACCURATE" in item else "ACCURATE" for item in labels]
                contains_empty = any(not label for label in new_labels)
                if contains_empty:
                    print("contains empty:", contains_empty, line)
                if not contains_empty:
                    new_file.write(json.dumps({
                        "question_id": question_id,
                        "image": image,
                        "question": question,
                        "response": response,
                        "sentences": sentences,
                        "new_labels": new_labels,
                        "cat_labels": labels,
                        "indices": indices,
                        "old_sentences": old_sentences,
                        "old_labels": old_labels
                    }) + "\n")
                    idx += 1
                    new_file.flush()
                # print(new_labels)


                # merged_response = " ".join(sentences)
                # print(response == merged_response)
                # for ele in line['annotations']:
                #     sentence = ele['text'].strip()
                #     label = ele['label'].strip()
                #     sentences.append(sentence)
                #     labels.append(label)


                # new_file.write(json.dumps({
                #     "question_id": question_id,
                #     "image": image,
                #     "question": text,
                #     "response": response,
                #     "sentences": sentences,
                #     "labels": labels,
                # }) + "\n")
                # idx += 1
                # new_file.flush()

    new_file.close()
    print(total_cases, total_inner_fails, total_outer_fails, total_successes)
