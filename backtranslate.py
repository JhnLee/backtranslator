import torch
import nltk
from tqdm import tqdm, trange


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


def split_sentences(contents, max_len):
    """Split paragraph to sentences
    Adapted from UDA official code
    """
    new_contents = []
    doc_len = []

    for i in range(len(contents)):
        contents[i] = contents[i].strip()
        if isinstance(contents[i], bytes):
            contents[i] = contents[i].decode("utf-8")
        sent_list = nltk.tokenize.sent_tokenize(contents[i])

        has_long = False
        if i % 100 == 0:
            print("splitting sentence {:d}".format(i))
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
                    sent_split = split_sent_by_punc(sent, split_punc, offset)
                    new_sent_list += sent_split
            sent_list = new_sent_list
            if not has_long:
                break

        contents[i] = None
        doc_len += [len(sent_list)]

        for st in sent_list:
            new_contents += [st]
    return new_contents, doc_len


def backtranslate(args, texts):
    # Load translation model from pytorch hub
    src2tgt = torch.hub.load(
        "pytorch/fairseq", args.src2tgt_model, tokenizer="moses", bpe="fastbpe"
    ).to(args.device)
    tgt2src = torch.hub.load(
        "pytorch/fairseq", args.tgt2src_model, tokenizer="moses", bpe="fastbpe"
    ).to(args.device)

    # TODO: Split workers for multi-gpu translate
    # if args.n_gpu > 1:
    #     src2tgt = torch.nn.DataParallel(src2tgt)
    #     tgt2src = torch.nn.DataParallel(tgt2src)

    splited_text, original_lens = split_sentences(texts, args.max_len)

    iterator = tqdm(range(len(splited_text) // args.batch_size + 1), desc="Iteration")

    back_translated_sents = []
    for i in iterator:
        start_idx = i * args.batch_size
        batch = splited_text[start_idx : start_idx + args.batch_size]
        translated_data = src2tgt.translate(
            batch,
            sampling_topk=args.sampling_topk,
            sampling_topp=args.sampling_topp,
            sampling=args.sampling,
            temperature=args.temperature,
            beam_size=args.beam_size,
        )
        backtranslated_data = tgt2src.translate(
            translated_data,
            sampling_topk=args.sampling_topk,
            sampling_topp=args.sampling_topp,
            sampling=args.sampling,
            temperature=args.temperature,
            beam_size=args.beam_size,
        )
        back_translated_sents += backtranslated_data

    back_translated_docs = []
    count = 0
    for i, n_sent in enumerate(original_lens):
        doc = []
        for _ in range(n_sent):
            doc.append(back_translated_sents[count].strip())
            count += 1
        back_translated_docs.append(" ".join(doc))

    return back_translated_docs
