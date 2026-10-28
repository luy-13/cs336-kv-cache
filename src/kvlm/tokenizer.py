import json
import pickle
import regex as re
from typing import Iterable, Iterator

# GPT-2 预分词正则表达式
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def merge_tuple(token_tuple: tuple[bytes, ...], best_pair: tuple[bytes, bytes]) -> tuple[bytes, ...]:
    """
    辅助函数：将元组中所有连续出现的 best_pair=(A, B) 替换为合并后的字节 (A + B)
    """
    A, B = best_pair
    new_tuple = []
    i = 0
    while i < len(token_tuple):
        if i < len(token_tuple) - 1 and token_tuple[i] == A and token_tuple[i + 1] == B:
            new_tuple.append(A + B)
            i += 2
        else:
            new_tuple.append(token_tuple[i])
            i += 1
    return tuple(new_tuple)


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = dict(vocab)
        self.merges = list(merges)
        self.special_tokens = special_tokens or []

        # 检查并追加尚未在 vocab 中的特殊 Token
        existing_vocab_bytes = set(self.vocab.values())
        next_id = max(self.vocab.keys(), default=-1) + 1 if self.vocab else 0
        for st in self.special_tokens:
            st_bytes = st.encode("utf-8")
            if st_bytes not in existing_vocab_bytes:
                self.vocab[next_id] = st_bytes
                existing_vocab_bytes.add(st_bytes)
                next_id += 1

        # 构建快速反向查找映射字典
        self.bytes_to_id = {v: k for k, v in self.vocab.items()}
        self.merge_ranks = {pair: i for i, pair in enumerate(self.merges)}

    @classmethod
    def from_files(
        cls, 
        vocab_filepath: str, 
        merges_filepath: str, 
        special_tokens: list[str] | None = None
    ) -> "Tokenizer":
        """
        从序列化文件中加载词表和合并规则并实例化 Tokenizer
        """
        # 读取词表 (支持 json 或 pickle)
        if vocab_filepath.endswith(".pickle") or vocab_filepath.endswith(".pkl"):
            with open(vocab_filepath, "rb") as f:
                vocab = pickle.load(f)
        else:
            with open(vocab_filepath, "r", encoding="utf-8") as f:
                raw_vocab = json.load(f)
                vocab = {}
                for k, v in raw_vocab.items():
                    token_id = int(k)
                    if isinstance(v, str):
                        token_bytes = v.encode("latin-1")  # 或 utf-8
                    else:
                        token_bytes = bytes(v)
                    vocab[token_id] = token_bytes

        # 读取合并规则 (支持 json 或 pickle/txt)
        if merges_filepath.endswith(".pickle") or merges_filepath.endswith(".pkl"):
            with open(merges_filepath, "rb") as f:
                merges = pickle.load(f)
        else:
            with open(merges_filepath, "r", encoding="utf-8") as f:
                raw_merges = json.load(f)
                merges = []
                for pair in raw_merges:
                    p1 = pair.encode("latin-1") if isinstance(pair, str) else bytes(pair)
                    p2 = pair[1].encode("latin-1") if isinstance(pair[1], str) else bytes(pair[1])
                    merges.append((p1, p2))

        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    def encode(self, text: str) -> list[int]:
        """
        将纯文本编码为 Token ID 列表
        """
        if not text:
            return []

        # 特殊 Token 隔离切分（按长度从长到短排列，避免短 token 抢先匹配）
        if self.special_tokens:
            ordered = sorted(self.special_tokens, key=len, reverse=True)
            pattern = f"({'|'.join(re.escape(st) for st in ordered)})"
            chunks = re.split(pattern, text)
        else:
            chunks = [text]

        ids = []
        special_set = set(self.special_tokens)

        for chunk in chunks:
            if not chunk:
                continue

            # 特殊 Token 直接转 ID
            if chunk in special_set:
                ids.append(self.bytes_to_id[chunk.encode("utf-8")])
                continue

            # 普通文本按正则预分词并规约合并
            for match in re.finditer(PAT, chunk):
                s = match.group(0)
                token_tuple = tuple(bytes([b]) for b in s.encode("utf-8"))

                while len(token_tuple) > 1:
                    pairs = [
                        (token_tuple[i], token_tuple[i + 1])
                        for i in range(len(token_tuple) - 1)
                        if (token_tuple[i], token_tuple[i + 1]) in self.merge_ranks
                    ]
                    if not pairs:
                        break
                    best_pair = min(pairs, key=lambda p: self.merge_ranks[p])
                    token_tuple = merge_tuple(token_tuple, best_pair)

                for token_bytes in token_tuple:
                    ids.append(self.bytes_to_id[token_bytes])

        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        对字符串迭代器进行延迟编码，逐个产出 Token ID
        """
        for chunk in iterable:
            for token_id in self.encode(chunk):
                yield token_id

    def decode(self, ids: list[int]) -> str:
        """
        将 Token ID 列表解码还原为字符串（遇到非有效 UTF-8 自动用 U+FFFD 替换）
        """
        raw_bytes = b"".join(self.vocab[i] for i in ids)
        return raw_bytes.decode("utf-8", errors="replace")