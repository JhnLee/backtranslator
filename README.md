# Back-translator
The back-translation module using fairseq translation models available in PyTorch Hub

## Requirements
torch, nltk, tqdm, cython, fastBPE

Make sure to download the nltk's punkt tokenizer  
```
$ python -c "import nltk; nltk.download('punkt')"
```

## Example Usage
Load pretrained NMT models [made by fairseq](https://github.com/pytorch/fairseq/tree/master/examples/translation)  
``` 
src2tgt = "transformer.wmt19.en-de.single_model"
tgt2src = "transformer.wmt19.de-en.single_model" 
# Tokenizer and BPE differ between the models
tokenizer = "moses"
bpe = "fastbpe" 
device = "cuda"

bt = BackTranslator(src2tgt, tgt2src, tokenizer, bpe, device)
```

Backtranslate one sentence
```
sample_sentence = "Python is an interpreted, high-level, general-purpose programming language."
bt.backtranslate(sample_sentence)

# 'Python is an interpreted, sophisticated, universal programming language.'
```

You can also use multiple documents (The length of the documents does not have any limitation)
```
sample_doc = ['Python is dynamically typed and garbage-collected.', 
              'It supports multiple programming paradigms, including structured (particularly, procedural), object-oriented, and functional programming.', 
              'Python is often described as a "batteries included" language due to its comprehensive standard library']
bt.backtranslate_docs(sample_doc)

# ['Python is typed dynamically and garbage collected.',
# 'It supports several programming paradigms, including structured (especially procedural), object oriented and functional programming.',
# "Python is often described as the 'Battery Language' because of the comprehensive standard library."]
```

By using `main.py`, you can train your own tsv document (to see the detail format of the tsv file, see the [example file](./data/imdb_sample.tsv))  
(Notice that the tsv file does not have to contain `label` columns.)  
```
# Example for using cpu
$ python main.py --data_dir=./data/imdb_sample.tsv --output_dir=./output/ --batch_size=32 

# for using single gpu (example for using only one gpu; gpu 1)
$ python main.py --data_dir=./data/imdb_sample.tsv --output_dir=./output/ --batch_size=64 --gpus 1

# for using multiple gpus (example for using two gpus; gpu 0 and 1)
$ python main.py --data_dir=./data/imdb_sample.tsv --output_dir=./output/ --batch_size=64 --gpus 0 1
```

## Reference
[google research UDA](https://github.com/google-research/uda)  
[fairseq NMT models](https://github.com/pytorch/fairseq/tree/master/examples/translation)
