import torch
import nltk
import logging
from tqdm import tqdm, trange


class BackTranslator:
    def __init__(self, src2tgt, tgt2src, tokenizer, bpe, device, visible_device=None):
        if visible_device is None and device != torch.device("cpu"):
            visible_device = int(device[-1])

        self.visible_device = visible_device
        self.src2tgt = (
            torch.hub.load("pytorch/fairseq", src2tgt, tokenizer=tokenizer, bpe=bpe)
            .to(device)
            .eval()
        )
        self.tgt2src = (
            torch.hub.load("pytorch/fairseq", tgt2src, tokenizer=tokenizer, bpe=bpe)
            .to(device)
            .eval()
        )

    @staticmethod
    def split_sent_by_punc(sent, punc, offset):
        """split sentence by the punctuation
        Adapted from UDA official code https://github.com/google-research/uda/blob/master/back_translate/split_paragraphs.py
        """
        sent_list = []
        start = 0
        while start < len(sent):
            if punc:
                pos = sent.find(punc, start + offset)
            else:
                pos = start + offset
            if pos != -1:
                sent_list += [sent[start : pos + 1]]
                start = pos + 1
            else:
                sent_list += [sent[start:]]
                break
        return sent_list

    def split_sentences(self, contents, max_len):
        """Split paragraph to sentences
        """
        new_contents = []
        doc_len = []

        for i in tqdm(
            range(len(contents)),
            desc="splits",
            disable=self.src2tgt.device.index not in [None, self.visible_device],
        ):
            contents[i] = contents[i].strip()
            if isinstance(contents[i], bytes):
                contents[i] = contents[i].decode("utf-8")
            sent_list = nltk.tokenize.sent_tokenize(contents[i])

            new_sent_list = []
            new_sent = ""
            for sent in sent_list:
                if len(new_sent) + len(sent) < max_len: 
                    new_sent += " " + sent
                elif len(sent) < max_len:
                    new_sent_list.append(new_sent.strip())
                    new_sent = sent
                else:
                    for split_punc in [".", ";", ",", " ", ""]:
                        if split_punc == " " or not split_punc:
                            offset = 100
                        else:
                            offset = 5
                        sent_split = BackTranslator.split_sent_by_punc(sent, split_punc, offset)
                        if all(len(s) < max_len for s in sent_split):
                            break
                    for sent_ in sent_split:
                        if len(new_sent) + len(sent_) < max_len: 
                            new_sent += " " + sent_
                        else:
                            new_sent_list.append(new_sent.strip())
                            new_sent = sent_
            if new_sent:
                new_sent_list.append(new_sent.strip())

            sent_list = new_sent_list

            contents[i] = None
            doc_len.append(len(sent_list))

            for st in sent_list:
                new_contents.append(st)
        return new_contents, doc_len

    def backtranslate(
        self,
        sent,
        sampling_topk=-1,
        sampling_topp=-1.0,
        sampling=True,
        temperature=0.9,
        beam_size=1,
    ):
        with torch.no_grad():
            translated_data = self.src2tgt.translate(
                sent,
                sampling_topk=sampling_topk,
                sampling_topp=sampling_topp,
                sampling=sampling,
                temperature=temperature,
                beam_size=beam_size,
            )
            backtranslated_data = self.tgt2src.translate(
                translated_data,
                sampling_topk=sampling_topk,
                sampling_topp=sampling_topp,
                sampling=sampling,
                temperature=temperature,
                beam_size=beam_size,
            )
            return backtranslated_data

    def backtranslate_docs(
        self,
        doc,
        max_len=300,
        batch_size=64,
        sampling_topk=-1,
        sampling_topp=-1.0,
        sampling=True,
        temperature=0.9,
        beam_size=1,
    ):
        splitted_text, original_lens = self.split_sentences(doc, max_len)

        iterator = tqdm(
            range(len(splitted_text) // batch_size + 1),
            desc="Iteration",
            disable=self.src2tgt.device.index not in [None, self.visible_device],
        )

        back_translated_sents = []
        for i in iterator:
            start_idx = i * batch_size
            original_sents = splitted_text[start_idx : start_idx + batch_size]
            back_translated_sents += self.backtranslate(
                original_sents, sampling_topk, sampling_topp, sampling, temperature, beam_size
            )

        back_translated_docs = []
        count = 0
        for i, n_sent in enumerate(original_lens):
            doc = []
            for _ in range(n_sent):
                doc.append(back_translated_sents[count].strip())
                count += 1
            back_translated_docs.append(" ".join(doc))

        return back_translated_docs
