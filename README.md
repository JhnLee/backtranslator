# Back-translator
The back-translation module using fairseq translation models available in PyTorch Hub

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

You can also use multiple documents
```
sample_doc = ['Python is dynamically typed and garbage-collected.', 
              'It supports multiple programming paradigms, including structured (particularly, procedural), object-oriented, and functional programming.', 
              'Python is often described as a "batteries included" language due to its comprehensive standard library']
bt.backtranslate_docs(sample_doc)

# ['Python is typed dynamically and garbage collected.',
# 'It supports several programming paradigms, including structured (especially procedural), object oriented and functional programming.',
# "Python is often described as the 'Battery Language' because of the comprehensive standard library."]
```

## Reference
[fairseq NMT models](https://github.com/pytorch/fairseq/tree/master/examples/translation)
