import torch
import nltk
from tqdm import tqdm, trange


class BackTranslator:
    def __init__(self, src2tgt, tgt2src, tokenizer, bpe, device):
        self.src2tgt = torch.hub.load("pytorch/fairseq", src2tgt, tokenizer=tokenizer, bpe=bpe).to(
            device
        )
        self.tgt2src = torch.hub.load("pytorch/fairseq", tgt2src, tokenizer=tokenizer, bpe=bpe).to(
            device
        )

        self.src2tgt.eval()
        self.tgt2src.eval()

    @staticmethod
    def split_sent_by_punc(sent, punc, offset):
        """split sentence by the punctuation
        Adapted from UDA official code
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
        Adapted from UDA official code
        """
        new_contents = []
        doc_len = []

        for i in tqdm(
            range(len(contents)), desc="splits", disable=self.src2tgt.device.index not in [None, 0]
        ):
            contents[i] = contents[i].strip()
            if isinstance(contents[i], bytes):
                contents[i] = contents[i].decode("utf-8")
            sent_list = nltk.tokenize.sent_tokenize(contents[i])

            has_long = False
            for split_punc in [".", ";", ",", " ", ""]:
                if split_punc == " " or not split_punc:
                    offset = 100
                else:
                    offset = 5
                has_long = False
                new_sent_list = []
                for sent in sent_list:
                    if len(sent) < max_len:
                        new_sent_list += [sent]
                    else:
                        has_long = True
                        sent_split = BackTranslator.split_sent_by_punc(sent, split_punc, offset)
                        new_sent_list += sent_split
                sent_list = new_sent_list
                if not has_long:
                    break

            contents[i] = None
            doc_len += [len(sent_list)]

            for st in sent_list:
                new_contents += [st]
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
        splited_text, original_lens = self.split_sentences(doc, max_len)

        iterator = tqdm(
            range(len(splited_text) // batch_size + 1),
            desc="Iteration",
            disable=self.src2tgt.device.index not in [None, 0],
        )

        back_translated_sents = []
        for i in iterator:
            start_idx = i * batch_size
            batch = splited_text[start_idx : start_idx + batch_size]
            back_translated_sents += self.backtranslate(
                batch, sampling_topk, sampling_topp, sampling, temperature, beam_size
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
