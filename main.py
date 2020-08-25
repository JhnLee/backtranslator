import torch
import os
import csv
import logging
import argparse
from backtranslate import BackTranslator

logger = logging.getLogger(__name__)


def clean_for_imdb(st):
    """clean text for IMDB datasets
    Adapted from UDA official code
    """
    st = st.replace("<br />", " ")
    st = st.replace("&quot;", '"')
    st = st.replace("<p>", " ")
    if "<a href=" in st:
        while "<a href=" in st:
            start_pos = st.find("<a href=")
            end_pos = st.find(">", start_pos)
            if end_pos != -1:
                st = st[:start_pos] + st[end_pos + 1 :]
            else:
                print("incomplete href")
                print("before {}".format(st))
                st = st[:start_pos] + st[start_pos + len("<a href=")]
                print("after {}".format(st))

        st = st.replace("</a>", "")
    st = st.replace("\\n", " ")
    return st


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data file (a text file).",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the backtranslated file will be saved.",
    )
    parser.add_argument(
        "--src2tgt_model",
        default="transformer.wmt19.en-de.single_model",
        type=str,
        help="The src2tgt translation model to use(from English to target language). Refer to https://github.com/pytorch/fairseq/tree/master/examples/translation.",
    )
    parser.add_argument(
        "--tgt2src_model",
        default="transformer.wmt19.de-en.single_model",
        type=str,
        help="The tgt2src translation model to use(from English to target language). Refer to https://github.com/pytorch/fairseq/tree/master/examples/translation.",
    )
    parser.add_argument(
        "--tokenizer", default="moses", type=str, help="Tokenizer for fairseq hub model",
    )
    parser.add_argument(
        "--bpe", default="fastbpe", type=str, help="BPE for fairseq hub model",
    )
    parser.add_argument(
        "--batch_size", type=int, required=True, help="Batch size",
    )
    parser.add_argument(
        "--max_len", default=300, type=int,
    )
    parser.add_argument(
        "--sampling_topk", default=-1, type=int,
    )
    parser.add_argument(
        "--sampling_topp", default=-1.0, type=float,
    )
    parser.add_argument(
        "--beam_size", default=1, type=int,
    )
    parser.add_argument(
        "--sampling", default=True,
    )
    parser.add_argument(
        "--temperature", default=0.9, type=float,
    )

    parser.add_argument("--no_cuda", action="store_true", help="Not to use CUDA")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    data = []
    with open(args.data_dir, encoding="utf-8") as f:
        for line in csv.reader(f, delimiter="\t"):
            data.append(line)
    text, labels = list(zip(*data[1:]))

    if isinstance(text, tuple):
        text = list(text)

    if "imdb" in args.data_dir or "IMDB" in args.data_dir:
        text = [clean_for_imdb(t) for t in text]

    print("Do back-translation for {} sentences".format(len(text)))

    bt = BackTranslator(
        args.src2tgt_model,
        args.tgt2src_model,
        tokenizer=args.tokenizer,
        bpe=args.bpe,
        device=args.device,
    )

    back_translated_docs = bt.backtranslate_docs(
        doc=text,
        max_len=args.max_len,
        batch_size=args.batch_size,
        sampling_topk=args.sampling_topk,
        sampling_topp=args.sampling_topp,
        sampling=args.sampling,
        temperature=args.temperature,
        beam_size=args.beam_size,
    )
    assert len(labels) == len(back_translated_docs)

    output_file_name = "bt_" + os.path.basename(args.data_dir)
    output_dir = os.path.join(args.output_dir, output_file_name)
    with open(output_dir, "wt") as f:
        tsv_writer = csv.writer(f, delimiter="\t")
        tsv_writer.writerow(data[0])
        for line, labels in zip(back_translated_docs, labels):
            tsv_writer.writerow([line, labels])

    print("Data saved as {}".format(output_dir))

