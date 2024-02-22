import argparse
from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation_qing import Chat, CONV_VISION_minigptv2
import numpy as np
import json
import os
import math
from tqdm import tqdm
import pickle

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def main(args):
    # ========================================
    #             Model Initialization
    # ========================================

    print('Initializing model')
    cfg = Config(args)
    model_config = cfg.model_cfg

    print(model_config)
    model_config.device_8bit = args.gpu_id
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))
    vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
    chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id))
    print('Model Initialization Finished')

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    responses = {}
    # idxs = ['COCO_val2014_000000193245', 'COCO_val2014_000000086848']
    count = 0
    for line in tqdm(questions):

        if count == 5:
            break

        idx = line["question_id"]

        # if idx not in idxs:
        #     break

        # Set Chat
        chat_state = CONV_VISION_minigptv2.copy()
        image_file = line['image']
        image_path = os.path.join(args.image_folder, image_file)
        img_list = []
        llm_message = chat.upload_img(image_path, chat_state, img_list)

        # prompt
        question = line['question']
        response = line['response']
        sentences = line['sentences']
        user_message = question + "<QuestoRes>" + response
        chat.ask(user_message, chat_state)
        chat.encode_img(img_list)

        # get answer
        # logprobs = chat.answer(conv=chat_state,
        #                       img_list=img_list,
        #                       num_beams=args.num_beams,
        #                       temperature=args.temperature,
        #                       max_new_tokens=10,
        #                       max_length=2000,
        #                       do_sample=False,
        #                       top_k=args.top_k,
        #                       top_p=args.top_p,
        #                       output_hidden_states=True,
        #                       return_dict_in_generate=True,
        #                       use_cache=True,
        #                       output_scores=True
        #                       )

        logprobs = chat.answer(conv=chat_state,
                               img_list=img_list,
                               output_hidden_states=True)

        output = {"question_id": idx,
                  "prompts": question,
                  "text": response,
                  "sentences": sentences,
                  "logprobs": logprobs
                  }

        responses[idx] = output
        count += 1

    with open(answers_file, 'wb') as file:
        pickle.dump(responses, file)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg_path", type=str, default="eval_configs/minigptv2_eval.yaml")
    # parser.add_argument("--img_path", type=str, default="data/val2014")
    parser.add_argument("--gpu_id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--temperature", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--image-folder", type=str, default="data/val2014")
    parser.add_argument("--question-file", type=str, default="data/synthetic_data_from_M_HalDetect.json")
    parser.add_argument("--answers-file", type=str, default="data/answer_synthetic_data_from_M_HalDetect.bin")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--options", nargs="+", help="override some settings in the used config, the key-value pair " "in xxx=yyy format will be merged into config file (deprecate), " "change to --cfg-options instead.",)
    args = parser.parse_args()
    main(args)
