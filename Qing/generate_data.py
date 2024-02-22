import numpy as np
import json



if __name__ == "__main__":

    new_file_name = "data/synthetic_train_data_from_M_HalDetect.json"
    new_file = open(new_file_name, "w")
    images = []
    ori_file_name = 'data/train_raw.json'
    with open(ori_file_name, 'r') as f:
        for file in f.readlines():
            dic = json.loads(file)
            for line in dic:
                text = line['question'].strip().replace("<image>", "").replace("\n", "")
                response = line['response'].strip()
                image = line['image'].strip()
                question_id = line['image'].split(".")[0]
                sentences = []
                labels = []
                for ele in line['annotations']:
                    sentence = ele['text'].strip()
                    label = ele['label'].strip()
                    sentences.append(sentence)
                    labels.append(label)

                new_file.write(json.dumps({
                    "question_id": question_id,
                    "image": image,
                    "question": text,
                    "response": response,
                    "sentences": sentences,
                    "labels": labels,
                }) + "\n")

                new_file.flush()

    new_file.close()