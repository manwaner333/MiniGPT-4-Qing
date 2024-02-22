import argparse
from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat, CONV_VISION_minigptv2
import numpy as np
# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg_path", type=str, default="eval_configs/minigptv2_eval.yaml")
    parser.add_argument("--img_path", type=str, default="")
    parser.add_argument("--gpu_id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--temperature", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
             "in xxx=yyy format will be merged into config file (deprecate), "
             "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args


def main():
    # ========================================
    #             Model Initialization
    # ========================================

    print('Initializing model')
    args = parse_args()
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

    # upload image
    chat_state = CONV_VISION_minigptv2.copy()
    img_list = []

    args.img_path = "view.jpg"
    for i in range(1):
        llm_message = chat.upload_img(args.img_path, chat_state, img_list)
        # print(llm_message)

    # ask a question
    user_message = "Describe the image."
    # user_message = "Provide a concise description of the given image."
    chat.ask(user_message, chat_state)
    chat.encode_img(img_list)

    # get answer
    llm_message = chat.answer(conv=chat_state,
                              img_list=img_list,
                              num_beams=args.num_beams,
                              temperature=args.temperature,
                              max_new_tokens=10,
                              max_length=2000,
                              do_sample=False,
                              top_k=args.top_k,
                              top_p=args.top_p,
                              output_hidden_states=True,
                              return_dict_in_generate=True,
                              use_cache=True,
                              output_scores=True
                              )
    message = llm_message[0]
    hidden_states = llm_message[2]

    np.save("Qing/result/prompt_1", hidden_states)

    qingli = 3
    # print(llm_message)


if __name__ == "__main__":
    main()
