from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import unicodedata
import os
import logging
import tempfile
import shutil
import json

from urllib.parse import urlparse
from pathlib import Path
from typing import Optional, Tuple, Union, IO, Callable, Set
from hashlib import sha256
from functools import wraps
from tqdm import tqdm

import boto3
from botocore.exceptions import ClientError
import requests


logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', 
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


PYTORCH_PRETRAINED_BERT_CACHE = Path(os.getenv('PYTORCH_PRETRAINED_BERT_CACHE',
                                               Path.home() / '.pytorch_pretrained_bert'))


PRETRAINED_VOCAB_ARCHIVE_MAP = {
    'bert-base-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt",
    'bert-large-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-vocab.txt",
    'bert-base-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-vocab.txt",
    'bert-large-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-vocab.txt",
    'bert-base-multilingual-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased-vocab.txt",
    'bert-base-multilingual-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased-vocab.txt",
    'bert-base-chinese': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-vocab.txt",
}
VOCAB_NAME = 'vocab.txt'


def load_vocab(vocab_file):
    vocab = collections.OrderedDict()
    index = 0
    with open(vocab_file, "r", encoding="UTF-8") as reader:
        while True:
            token = reader.readline()
            if not token:
                break
            token = token.strip()
            vocab[token] = index
            index += 1
    return vocab


def whitespace_tokenize(text):
    text = text.strip()
    if not text:
        return []

    tokens = text.split()
    return tokens

def _is_whitespace(char):
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False

def _is_control(char):
    if char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False


def _is_punctuation(char):
    cp = ord(char)
    if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False

def s3_request(func: Callable):
    @wraps(func)
    def wrapper(url: str, *args, **kwargs):
        try:
            return func(url, *args, **kwargs)
        except ClientError as exc:
            if int(exc.response["Error"]["Code"]) == 404:
                raise FileNotFoundError("file {} not found".format(url))
            else:
                raise

    return wrapper

def split_s3_path(url: str) -> Tuple[str, str]:
    parsed = urlparse(url)
    if not parsed.netloc or not parsed.path:
        raise ValueError("bad s3 path {}".format(url))
    bucket_name = parsed.netloc
    s3_path = parsed.path
    if s3_path.startswith("/"):
        s3_path = s3_path[1:]
    return bucket_name, s3_path

@s3_request
def s3_etag(url: str) -> Optional[str]:
    """Check ETag on S3 object."""
    s3_resource = boto3.resource("s3")
    bucket_name, s3_path = split_s3_path(url)
    s3_object = s3_resource.Object(bucket_name, s3_path)
    return s3_object.e_tag

def url_to_filename(url: str, etag: str = None) -> str:
    url_bytes = url.encode('utf-8')
    url_hash = sha256(url_bytes)
    filename = url_hash.hexdigest()

    if etag:
        etag_bytes = etag.encode('utf-8')
        etag_hash = sha256(etag_bytes)
        filename += '.' + etag_hash.hexdigest()

    return filename

@s3_request
def s3_get(url: str, temp_file: IO) -> None:
    s3_resource = boto3.resource("s3")
    bucket_name, s3_path = split_s3_path(url)
    s3_resource.Bucket(bucket_name).download_fileobj(s3_path, temp_file)

def http_get(url: str, temp_file: IO) -> None:
    req = requests.get(url, stream=True)
    content_length = req.headers.get('Content-Length')
    total = int(content_length) if content_length is not None else None
    progress = tqdm(unit="B", total=total)
    for chunk in req.iter_content(chunk_size=1024):
        if chunk: 
            progress.update(len(chunk))
            temp_file.write(chunk)
    progress.close()


def get_from_cache(url : str, cache_dir : str = None) -> str:
    if cache_dir is None:
        cache_dir = PYTORCH_PRETRAINED_BERT_CACHE
    os.makedirs(cache_dir, exist_ok=True)
    if url.startswith("s3://"):
        etag = s3_etag(url)
    else:
        response = requests.head(url, allow_redirects=True)
        if response.status_code != 200:
            raise IOError()
        etag = response.headers.get('ETag')
    filename = url_to_filename(url, etag)
    cache_path = os.path.join(cache_dir, filename)

    if not os.path.exists(cache_path):
        with tempfile.NamedTemporaryFile() as temp_file:
            logger.info("%s not found in cache, downloading to %s", url, temp_file.name)
            if url.startswith("s3://"):
                s3_get(url, temp_file)
            else:
                http_get(url, temp_file)
            temp_file.flush()
            temp_file.seek(0)
            logger.info("copying %s to cache at %s", temp_file.name, cache_path)
            with open(cache_path, 'wb') as cache_file:
                shutil.copyfileobj(temp_file, cache_file)
            logger.info("creating metadata file for %s", cache_path)
            meta = {'url': url, 'etag': etag}
            meta_path = cache_path + '.json'
            with open(meta_path, 'w') as meta_file:
                json.dump(meta, meta_file)
            logger.info("removing temp file %s", temp_file.name)

    return cache_path


def cached_path(url_or_filename: Union[str, Path], cache_dir : str = None) -> str:
    if cache_dir is None:
        cache_dir = PYTORCH_PRETRAINED_BERT_CACHE
    if isinstance(url_or_filename, Path):
        url_or_filename = str(url_or_filename)

    parsed = urlparse(url_or_filename)

    if parsed.scheme in ('http', 'https', "s3"):
        return get_from_cache(url_or_filename, cache_dir)
    
    elif os.path.exists(url_or_filename):
        return url_or_filename
    elif parsed.scheme == "":
        raise FileNotFoundError()
    else:
        raise ValueError("")


class BasicTokenizer():

    def __init__(self, do_lower_case = True):
        self.do_lower_case = do_lower_case

    def tokenize(self, text):

        text = self._clean_text(text)
        text = self._tokenize_chars(text)
        orig_tokens = whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            if self.do_lower_case:
                token = token.lower()
                token = self._run_strip_accents(token)
            split_tokens.extend(self._run_split_on_punc(token))

        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens

    def _run_strip_accents(self, text):
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)

    def _run_split_on_punc(self, text):
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1
        return ["".join(x) for x in output]

    
    def _tokenize_chars(self, text):
        output = []
        for char in text:
            cp = ord(char)
            if self._check(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    def _check(self, cp):
        if ((cp >= 0x4E00 and cp <= 0x9FFF) or  
            (cp >= 0x3400 and cp <= 0x4DBF) or  
            (cp >= 0x20000 and cp <= 0x2A6DF) or  
            (cp >= 0x2A700 and cp <= 0x2B73F) or  
            (cp >= 0x2B740 and cp <= 0x2B81F) or  
            (cp >= 0x2B820 and cp <= 0x2CEAF) or
            (cp >= 0xF900 and cp <= 0xFAFF) or  
            (cp >= 0x2F800 and cp <= 0x2FA1F)):  
            return True
    
        return False

    def _clean_text(self, text):
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xfffd or _is_control(char):
                continue
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        
        return "".join(output)

class WordpieceTokenizer():

    def __init__(self, vocab, unk_token = "[UNK]", max_input_chars_per_word = 100):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text):
        output_tokens = []
        for token in whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens


class BertTokenizer():

    def __init__(self, vocab_file, do_lower_case = True):
        
        assert os.path.isfile(vocab_file), "Error: vocab file not founded"

        self.vocab = load_vocab(vocab_file)
        self.ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])
        self.basic_tokenizer = BasicTokenizer(do_lower_case = do_lower_case)
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab = self.vocab)

    def tokenize(self, text):
        split_tokens = []
        for token in self.basic_tokenizer.tokenize(text):
            for sub_token in self.wordpiece_tokenizer.tokenize(token):
                split_tokens.append(sub_token)
        return split_tokens

    def convert_tokens_to_ids(self, tokens):
        ids = []
        for token in tokens:
            ids.append(self.vocab[token])
        return ids

    def convert_ids_to_tokens(self, ids):
        tokens = []
        for i in ids:
            tokens.append(self.ids_to_tokens[i])
        return tokens

    @classmethod
    def from_pretrained(cls, pretrained_model_name, cache_dir = None, *inputs, **kwargs):
        if pretrained_model_name in PRETRAINED_VOCAB_ARCHIVE_MAP:
            vocab_file = PRETRAINED_VOCAB_ARCHIVE_MAP[pretrained_model_name]
        else:
            vocab_file = pretrained_model_name
        
        if os.path.isdir(vocab_file):
            vocab_file = os.path.join(vocab_file, VOCAB_NAME)
        
        try:
            resolved_vocab_file = cached_path(vocab_file, cache_dir = cache_dir)
        except FileNotFoundError:
            logger.error(
                "ERROR"
            )
            return None
        if resolved_vocab_file == vocab_file:
            logger.info("loading vocabulary file {}".format(vocab_file))
        else:
            logger.info("loading vocabulary file {} from cache at {}".format(
                vocab_file, resolved_vocab_file))
        tokenizer = cls(resolved_vocab_file, *inputs, **kwargs)
        return tokenizer


    