import argparse
import time
from threading import Thread
from PIL import Image

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer
from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer

import dataclasses
from enum import auto, Enum
from typing import List, Tuple, Any
import numpy as np
from scipy.stats import entropy
from minigpt4.common.registry import registry


class SeparatorStyle(Enum):
    """Different separator style."""
    SINGLE = auto()
    TWO = auto()


@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""
    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    # system_img: List[Image.Image] = []
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = "###"
    sep2: str = None

    skip_next: bool = False
    conv_id: Any = None

    def get_prompt(self):
        if self.sep_style == SeparatorStyle.SINGLE:
            ret = self.system + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + message + self.sep
                else:
                    ret += role
            return ret
        elif self.sep_style == SeparatorStyle.TWO:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + message + seps[i % 2]
                else:
                    ret += role
            return ret
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

    def append_message(self, role, message):
        self.messages.append([role, message])

    def to_gradio_chatbot(self):
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def copy(self):
        return Conversation(
            system=self.system,
            # system_img=self.system_img,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            conv_id=self.conv_id)

    def dict(self):
        return {
            "system": self.system,
            # "system_img": self.system_img,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "sep": self.sep,
            "sep2": self.sep2,
            "conv_id": self.conv_id,
        }


class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all(input_ids[:, -len(stop):] == stop).item():
                return True

        return False


CONV_VISION_Vicuna0 = Conversation(
    system="Give the following image: <Img>ImageContent</Img>. "
           "You will be able to see the image once I provide it to you. Please answer my questions.",
    roles=("Human: ", "Assistant: "),
    messages=[],
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)

CONV_VISION_LLama2 = Conversation(
    system="Give the following image: <Img>ImageContent</Img>. "
           "You will be able to see the image once I provide it to you. Please answer my questions.",
    roles=("<s>[INST] ", " [/INST] "),
    messages=[],
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="",
)

CONV_VISION_minigptv2 = Conversation(
    system="",
    roles=("<s>[INST] ", " [/INST]"),
    messages=[],
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="",
)

class Chat:
    def __init__(self, model, vis_processor, device='cuda:0', stopping_criteria=None):
        self.device = device
        self.model = model
        self.vis_processor = vis_processor

        if stopping_criteria is not None:
            self.stopping_criteria = stopping_criteria
        else:
            stop_words_ids = [torch.tensor([2]).to(self.device)]
            self.stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

    def ask(self, text, conv):
        if len(conv.messages) > 0 and conv.messages[-1][0] == conv.roles[0] \
                and conv.messages[-1][1][-6:] == '</Img>':  # last message is image.
            conv.messages[-1][1] = ' '.join([conv.messages[-1][1], text])
        else:
            conv.append_message(conv.roles[0], text)

    def answer_prepare(self, conv, img_list, max_new_tokens=300, max_length=2000, output_hidden_states=True, output_attentions=True):
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        embs, seg_tokens = self.model.get_context_emb(prompt, img_list)

        current_max_len = embs.shape[1] + max_new_tokens
        if current_max_len - max_length > 0:
            print('Warning: The number of tokens in current conversation exceeds the max length. '
                  'The model will not see the contexts outside the range.')
        begin_idx = max(0, current_max_len - max_length)
        embs = embs[:, begin_idx:]

        generation_kwargs = dict(
            inputs_embeds=embs,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions)

        return generation_kwargs, seg_tokens


    def answer(self, conv, img_list, sentences,  **kargs):
        generation_dict, seg_tokens = self.answer_prepare(conv, img_list, **kargs)
        img_len = img_list[0].shape[1]
        seg_tokens_1_len = seg_tokens[0].shape[1]
        seg_tokens_ques = seg_tokens[1]
        seg_tokens_ques_len = seg_tokens[1].shape[1]
        seg_tokens_res = seg_tokens[2]
        seg_tokens_res_len = seg_tokens[2].shape[1]
        ques_start_idx = seg_tokens_1_len + img_len + 3  # 3 对应的token:"<', Img', '>'
        ques_end_idx = seg_tokens_1_len + img_len + seg_tokens_ques_len - 1
        res_start_idx = seg_tokens_1_len + img_len + seg_tokens_ques_len
        res_end_idx = seg_tokens_1_len + img_len + seg_tokens_ques_len + seg_tokens_res_len - 4 - 1   # 4 对应的token: '[', '/', 'INST', ']'


        outputs = self.model_generate(**generation_dict)

        # hidden states
        hidden_states = outputs['hidden_states'][32][0]
        ques_hidden_states = hidden_states[ques_start_idx:ques_end_idx + 1, :]
        res_hidden_states = hidden_states[res_start_idx:res_end_idx + 1, :]

        # logit
        logits = outputs['logits']

        # attention
        attentions = outputs['attentions']

        shifted_logits = logits[:, ques_start_idx:res_end_idx, :]
        shifted_input_ids = torch.cat((seg_tokens[0], seg_tokens[1], seg_tokens[2]), 1)[:, ques_start_idx - img_len + 1:res_end_idx - img_len + 1]


        # Convert logits to probabilities
        log_probs = torch.nn.functional.log_softmax(shifted_logits, dim=-1)
        gathered_log_probs = torch.gather(log_probs, 2, shifted_input_ids.unsqueeze(-1)).squeeze(-1)
        gathered_log_probs = gathered_log_probs.detach().cpu().numpy()

        # Convert logits to entropies
        probs = torch.softmax(shifted_logits, dim=-1)[0]
        probs = probs.detach().cpu().numpy()
        entropies = 2 ** (entropy(probs, base=2, axis=-1))

        tokens = []
        token_logprobs = []
        token_entropies = []
        tokens_idx = []
        token_and_logprobs = []
        for t in range(shifted_input_ids.shape[1]):
            gen_tok_id = shifted_input_ids[:, t]
            gen_tok = self.model.llama_tokenizer.decode(gen_tok_id)
            lp = gathered_log_probs[:, t]
            entro = entropies[t]

            tokens.append(gen_tok)
            token_logprobs.append(lp)
            token_entropies.append(entro)
            token_and_logprobs.append([gen_tok, lp, entro])
            tokens_idx.append(gen_tok_id.detach().cpu().numpy().tolist())

        # 提取每个句子最后一个token的hidden state
        combined_attentions = {}
        combined_hidden_states = {}
        combined_token_logprobs = {}
        combined_token_entropies = {}
        ques_hidden_states = hidden_states[ques_end_idx:ques_end_idx + 1, :]
        combined_hidden_states["ques"] = ques_hidden_states.detach().cpu().numpy().tolist()
        combined_attentions['ques'] = attentions[31][0, :, ques_start_idx:ques_end_idx + 1, ques_start_idx:ques_end_idx + 1].detach().cpu().numpy().tolist()


        sentences_end = []
        sentences_end.append(ques_end_idx)
        start_idx = res_start_idx
        record = []
        record1 = []
        for sent_i, sentence in enumerate(sentences):
            # sentence exist in the passage, so we need to find where it is [i1, i2]
            sentence_tf = "".join(sentence.split(" "))
            xarr = [i for i in range(len(tokens))]
            for i1 in xarr:
                mystring = "".join(tokens[i1:])
                if sentence_tf not in mystring:
                    break
            i1 = i1 - 1
            for i2 in xarr[::-1]:
                mystring = "".join(tokens[i1:i2 + 1])
                if sentence_tf not in mystring:
                    break
            i2 = i2 + 1

            sentence_len = i2 - i1 + 1
            sentence_end = start_idx + sentence_len - 1
            hidden_state = hidden_states[sentence_end:sentence_end+1, :]
            attention = attentions[31][0, :, start_idx:sentence_end + 1, start_idx:sentence_end + 1].detach().cpu().numpy().tolist()
            combined_hidden_states[sent_i] = hidden_state.detach().cpu().numpy().tolist()
            combined_token_logprobs[sent_i] = token_logprobs[i1:i2+1]
            combined_token_entropies[sent_i] = token_entropies[i1:i2+1]
            combined_attentions[sent_i] = attention
            sentences_end.append(sentence_end)
            record.append([i1, i2])
            record1.append([start_idx, sentence_end])
            start_idx = sentence_end + 1

        # 提取每个句子每一个token的hidden state
        # combined_hidden_states = {}
        # combined_token_logprobs = {}
        # combined_token_entropies = {}
        # ques_hidden_states = hidden_states[ques_start_idx:ques_end_idx + 1, :]
        # combined_hidden_states["ques"] = ques_hidden_states.detach().cpu().numpy().tolist()
        #
        # sentences_end = []
        # sentences_end.append(ques_end_idx)
        # start_idx = res_start_idx
        # record = []
        # for sent_i, sentence in enumerate(sentences):
        #     # sentence exist in the passage, so we need to find where it is [i1, i2]
        #     sentence_tf = "".join(sentence.split(" "))
        #     xarr = [i for i in range(len(tokens))]
        #     for i1 in xarr:
        #         mystring = "".join(tokens[i1:])
        #         if sentence_tf not in mystring:
        #             break
        #     i1 = i1 - 1
        #     for i2 in xarr[::-1]:
        #         mystring = "".join(tokens[i1:i2 + 1])
        #         if sentence_tf not in mystring:
        #             break
        #     i2 = i2 + 1
        #
        #     sentence_len = i2 - i1 + 1
        #     sentence_end = start_idx + sentence_len - 1
        #     hidden_state = hidden_states[start_idx:sentence_end+1, :]
        #     combined_hidden_states[sent_i] = hidden_state.detach().cpu().numpy().tolist()
        #     combined_token_logprobs[sent_i] = token_logprobs[i1:i2+1]
        #     combined_token_entropies[sent_i] = token_entropies[i1:i2+1]
        #     sentences_end.append(sentence_end)
        #     record.append([start_idx, sentence_end])
        #     start_idx = sentence_end + 1

        outputs = {
            "tokens": tokens,
            "token_logprobs": token_logprobs,
            "token_entropies": token_entropies,
            "tokens_idx": tokens_idx,
            # "ques_hidden_states": ques_hidden_states.detach().cpu().numpy().tolist(),
            # "res_hidden_states": res_hidden_states.detach().cpu().numpy().tolist(),
            "combined_hidden_states": combined_hidden_states,
            "combined_token_logprobs": combined_token_logprobs,
            "combined_token_entropies": combined_token_entropies,
            "combined_attentions": combined_attentions,
            "token_and_logprobs": token_and_logprobs
        }

        conv.messages[-1][1] = ""

        return outputs


    def stream_answer(self, conv, img_list, **kargs):
        generation_kwargs = self.answer_prepare(conv, img_list, **kargs)
        streamer = TextIteratorStreamer(self.model.llama_tokenizer, skip_special_tokens=True)
        generation_kwargs['streamer'] = streamer
        thread = Thread(target=self.model_generate, kwargs=generation_kwargs)
        thread.start()
        return streamer

    def model_generate(self, *args, **kwargs):
        # for 8 bit and 16 bit compatibility
        with self.model.maybe_autocast():
            output = self.model.llama_model(*args, **kwargs)
        return output

    def encode_img(self, img_list):
        image = img_list[0]
        img_list.pop(0)
        if isinstance(image, str):  # is a image path
            raw_image = Image.open(image).convert('RGB')
            image = self.vis_processor(raw_image).unsqueeze(0).to(self.device)
        elif isinstance(image, Image.Image):
            raw_image = image
            image = self.vis_processor(raw_image).unsqueeze(0).to(self.device)
        elif isinstance(image, torch.Tensor):
            if len(image.shape) == 3:
                image = image.unsqueeze(0)
            image = image.to(self.device)

        image_emb, _ = self.model.encode_img(image)
        img_list.append(image_emb)

    def upload_img(self, image, conv, img_list):
        conv.append_message(conv.roles[0], "<Img><ImageHere></Img>")
        img_list.append(image)
        msg = "Received."

        return msg

