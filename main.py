import torch
import os
import csv
import pickle
import logging
import argparse
from torch.multiprocessing import set_start_method, Process, Queue
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
                logger.info("incomplete href")
                logger.info("before {}".format(st))
                st = st[:start_pos] + st[start_pos + len("<a href=")]
                logger.info("after {}".format(st))

        st = st.replace("</a>", "")
    st = st.replace("\\n", " ")
    return st


def translate(args, doc, gpu_num=None):
    if gpu_num is not None:
        device = torch.device("cuda:" + str(gpu_num))
        visible = args.gpus[0]
    else:
        device, visible = torch.device("cpu"), None
    bt = BackTranslator(
        args.src2tgt_model,
        args.tgt2src_model,
        tokenizer=args.tokenizer,
        bpe=args.bpe,
        device=device,
        visible_device=visible,
    )
    result = bt.backtranslate_docs(
        doc=doc,
        max_len=args.max_len,
        batch_size=args.batch_size,
        sampling_topk=args.sampling_topk,
        sampling_topp=args.sampling_topp,
        sampling=args.sampling,
        temperature=args.temperature,
        beam_size=args.beam_size,
        return_sentence_pair=args.return_sentence_pair,
    )
    return (gpu_num, result) if gpu_num is not None else result


def multi_translate(args, gpu_num, doc, queue):
    translated_doc = translate(args, doc, gpu_num)
    queue.put(translated_doc)


def main(args):
    if args.labels:
        data = []
        with open(args.data_dir, encoding="utf-8") as f:
            for line in csv.reader(f, delimiter="\t"):
                data.append(line)
            text, labels = list(zip(*data[1:]))
    else:
        text = []
        with open(args.data_dir, encoding="utf-8") as f:
            for line in f.readlines():
                text.append(line.strip())
        labels = None

    if isinstance(text, tuple):
        text = list(text)

    if "imdb" in args.data_dir or "IMDB" in args.data_dir:
        text = [clean_for_imdb(t) for t in text]

    logger.info("Do back-translation for {} sentences".format(len(text)))

    if args.gpus is not None and len(args.gpus) > 1:
        logger.info("Use Multiple GPUs: {}".format(", ".join([str(i) for i in args.gpus])))
        split_point = len(text) // len(args.gpus)

        text_splitted = []
        for gpu_id in args.gpus:
            text_splitted.append(text[gpu_id * split_point : (gpu_id + 1) * split_point])
            if gpu_id == len(args.gpus) - 1:
                text_splitted[-1] += text[(gpu_id + 1) * split_point :]
        assert sum(len(s) for s in text_splitted) == len(text)

        set_start_method("spawn")
        q = Queue()

        procs = []
        for i in range(len(args.gpus)):
            proc = Process(target=multi_translate, args=(args, i, text_splitted[i], q))
            procs.append(proc)
            proc.start()

        q_result = []
        for p in procs:
            q_result.append(q.get())

        back_translated_docs = []
        for doc_split in sorted(q_result):
            back_translated_docs += doc_split[1]

        q.close()
        q.join_thread()

        for proc in procs:
            proc.join()
    else:
        if args.gpus is not None:
            gpu = args.gpus[0]
            logger.info("Use only one GPU: {}".format(gpu))
            back_translated_docs = translate(args, text, args.gpus[0])[1]
        else:
            logger.info("Use cpu")
            back_translated_docs = translate(args, text)

    output_file_name = "bt_" + os.path.basename(args.data_dir)
    output_dir = os.path.join(args.output_dir, output_file_name)

    folder_name = os.path.dirname(output_dir)
    if not os.path.isdir(folder_name):
        os.makedirs(folder_name)

    if args.return_sentence_pair:
        # Save original sentence pair
        filename, ext = os.path.splitext(output_dir)
        with open(filename + ".pickle", "wb") as f:
            pickle.dump(back_translated_docs, f)

        # Save back-translated sentences
        bt_doc = [" ".join(list(zip(*d))[1]) for d in back_translated_docs]
        with open(output_dir, "wt") as f:
            if labels is not None:
                tsv_writer = csv.writer(f, delimiter="\t")
                tsv_writer.writerow(data[0])
                for line, label in zip(bt_doc, labels):
                    tsv_writer.writerow([line, label])
            else:
                for line in bt_doc:
                    f.write(line)
                    f.write('\n')

        # Save cross sentences
        new_back_translated_docs = []
        for doc in back_translated_docs:
            new_doc = []
            for j, sent in enumerate(doc):
                if j % 2 == 0:
                    new_doc.append(sent)
                else:
                    new_doc.append(sent[::-1])
            new_back_translated_docs.append(new_doc)
        new_docs1, new_docs2 = [], []
        for doc in new_back_translated_docs:
            n1, n2 = list(zip(*doc))
            new_docs1.append(" ".join(n1))
            new_docs2.append(" ".join(n2))
        
        filename, ext = os.path.splitext(output_dir)
        with open(filename + "_pair1" + ext, "wt") as f:
            if labels is not None:
                tsv_writer = csv.writer(f, delimiter="\t")
                tsv_writer.writerow(data[0])
                for line, label in zip(new_docs1, labels):
                    tsv_writer.writerow([line, label])
            else:
                for line in new_docs1:
                    f.write(line)
                    f.write('\n')
        with open(filename + "_pair2" + ext, "wt") as f:
            if labels is not None:
                tsv_writer = csv.writer(f, delimiter="\t")
                tsv_writer.writerow(data[0])
                for line, label in zip(new_docs2, labels):
                    tsv_writer.writerow([line, label])
            else:
                for line in new_docs2:
                    f.write(line)
                    f.write('\n')
    else:
        with open(output_dir, "wt") as f:
            if labels is not None:
                tsv_writer = csv.writer(f, delimiter="\t")
                tsv_writer.writerow(data[0])
                for line, label in zip(back_translated_docs, labels):
                    tsv_writer.writerow([line, label])
            else:
                for line in back_translated_docs:
                    f.write(line)
                    f.write('\n')

    logger.info("Translated documents are saved in {}".format(output_dir))


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
        "--labels",
        action="store_true",
        help="Whether to have label columns",
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
        "--tokenizer",
        default="moses",
        type=str,
        help="Tokenizer for fairseq hub model",
    )
    parser.add_argument(
        "--bpe",
        default="fastbpe",
        type=str,
        help="BPE for fairseq hub model",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        required=True,
        help="Batch size",
    )
    parser.add_argument(
        "--max_len",
        default=250,
        type=int,
    )
    parser.add_argument(
        "--sampling_topk",
        default=-1,
        type=int,
    )
    parser.add_argument(
        "--sampling_topp",
        default=-1.0,
        type=float,
    )
    parser.add_argument(
        "--beam_size",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--sampling",
        default=True,
    )
    parser.add_argument(
        "--temperature",
        default=0.9,
        type=float,
    )
    parser.add_argument(
        "--gpus",
        nargs="+",
        default=None,
        type=int,
    )
    parser.add_argument(
        "--return_sentence_pair",
        action="store_true",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Not to use CUDA")
    args = parser.parse_args()

    if args.gpus is not None:
        assert (
            len(args.gpus) <= torch.cuda.device_count()
        ), "The number of GPU used is more than you have"

    # Setup logger
    logging.basicConfig(
        format="%(asctime)s - %(name)s -  %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    main(args)
