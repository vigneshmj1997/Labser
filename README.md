
Labser is a library to calculate and use multilingual sentence embeddings.

## Dependencies
* Python 3.6
* [PyTorch 1.0](http://pytorch.org/)
* [NumPy](http://www.numpy.org/), tested with 1.15.4
* [Cython](https://pypi.org/project/Cython/), needed by Python wrapper of FastBPE, tested with 0.29.6
* [Faiss](https://github.com/facebookresearch/faiss), for fast similarity search and bitext mining
* [transliterate 1.10.2](https://pypi.org/project/transliterate), only used for Greek (`pip install transliterate`)
* [jieba 0.39](https://pypi.org/project/jieba/), Chinese segmenter (`pip install jieba`)
* [mecab 0.996](https://pypi.org/project/JapaneseTokenizer/), Japanese segmenter
* tokenization from the Moses encoder (installed automatically)
* [FastBPE](https://github.com/glample/fastBPE), fast C++ implementation of byte-pair encoding (installed automatically)

## Installation
* set the environment variable 'LASER' to the root of the installation, e.g.
  `export LASER="${HOME}/projects/laser"`
* download encoders from Amazon s3 by `bash ./install_models.sh`
* download third party software by `bash ./install_external_tools.sh`

