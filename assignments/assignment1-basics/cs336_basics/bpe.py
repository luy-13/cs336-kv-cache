import heapq
from collections import defaultdict

import regex as re

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
        # 检查是否匹配到要合并的字节对
        if i < len(token_tuple) - 1 and token_tuple[i] == A and token_tuple[i + 1] == B:
            new_tuple.append(A + B)
            i += 2  # 跳过被合并的两个元素
        else:
            new_tuple.append(token_tuple[i])
            i += 1  # 保持原样，推进 1 位

    return tuple(new_tuple)


class _DescPair:
    """堆排序辅助：频次相同时让字节序更大的 pair 先出堆（与参考实现的平局规则一致）"""

    __slots__ = ("pair",)

    def __init__(self, pair: tuple[bytes, bytes]) -> None:
        self.pair = pair

    def __lt__(self, other: "_DescPair") -> bool:
        return self.pair > other.pair


def _add_pairs(
    token_tuple: tuple[bytes, ...],
    weight: int,
    pair_counts: dict[tuple[bytes, bytes], int],
    pair_to_tokens: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]],
    heap: list[tuple[int, _DescPair]],
) -> None:
    """把某个 pre-token 贡献的相邻 pair 计入全局统计，并登记反向索引"""
    for i in range(len(token_tuple) - 1):
        pair = (token_tuple[i], token_tuple[i + 1])
        new_count = pair_counts.get(pair, 0) + weight
        pair_counts[pair] = new_count
        pair_to_tokens[pair].add(token_tuple)
        heapq.heappush(heap, (-new_count, _DescPair(pair)))


def _remove_pairs(
    token_tuple: tuple[bytes, ...],
    weight: int,
    pair_counts: dict[tuple[bytes, bytes], int],
    pair_to_tokens: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]],
    heap: list[tuple[int, _DescPair]],
) -> None:
    """把一个 pre-token 从全局 pair 统计中撤销，并清理反向索引"""
    for i in range(len(token_tuple) - 1):
        pair = (token_tuple[i], token_tuple[i + 1])
        new_count = pair_counts[pair] - weight
        if new_count > 0:
            pair_counts[pair] = new_count
            heapq.heappush(heap, (-new_count, _DescPair(pair)))
        else:
            pair_counts.pop(pair, None)
        holders = pair_to_tokens.get(pair)
        if holders is not None:
            holders.discard(token_tuple)
            if not holders:
                pair_to_tokens.pop(pair, None)


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    训练字节级 BPE Tokenizer

    与逐轮全量重扫的实现相比，这里维护 pair 频次的增量统计（pair -> 计数、
    pair -> 包含它的 pre-token 集合）以及一个惰性删除的大顶堆，使每次合并只需
    处理真的受影响的 pre-token。
    """
    vocab: dict[int, bytes] = {}
    merges: list[tuple[bytes, bytes]] = []

    # -------------------------------------------------------------
    # 步骤 1：初始化词表（Special Tokens + 256 基础字节）
    # -------------------------------------------------------------
    next_id = 0
    for st in special_tokens:
        vocab[next_id] = st.encode("utf-8")
        next_id += 1

    for b in range(256):
        vocab[next_id] = bytes([b])
        next_id += 1

    # -------------------------------------------------------------
    # 步骤 2：读取文本，使用特殊 Token 隔离并进行预分词
    # -------------------------------------------------------------
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    if special_tokens:
        ordered = sorted(special_tokens, key=len, reverse=True)
        pattern = "|".join(re.escape(st) for st in ordered)
        chunks = re.split(pattern, text)
    else:
        chunks = [text]

    pre_token_counts: dict[tuple[bytes, ...], int] = defaultdict(int)
    for chunk in chunks:
        if not chunk:
            continue
        for match in re.finditer(PAT, chunk):
            s = match.group(0)
            token_tuple = tuple(bytes([b]) for b in s.encode("utf-8"))
            pre_token_counts[token_tuple] += 1

    # -------------------------------------------------------------
    # 步骤 3 & 4：增量维护 pair 频次，迭代合并最高频字节对
    # -------------------------------------------------------------
    pair_counts: dict[tuple[bytes, bytes], int] = {}
    pair_to_tokens: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]] = defaultdict(set)
    heap: list[tuple[int, _DescPair]] = []

    for token_tuple, count in pre_token_counts.items():
        _add_pairs(token_tuple, count, pair_counts, pair_to_tokens, heap)

    while len(vocab) < vocab_size:
        # 3.1 从堆顶取出当前频次最高的 pair；跳过已被后续更新覆盖的过期堆项
        best_pair = None
        while heap:
            neg_count, wrapper = heap[0]
            pair = wrapper.pair
            if neg_count < 0 and pair_counts.get(pair, 0) == -neg_count:
                best_pair = pair
                break
            heapq.heappop(heap)

        # 如果没有可以合并的 pair，提前结束循环
        if best_pair is None:
            break

        # 3.2 记录合并规则，更新词表
        A, B = best_pair
        merges.append(best_pair)
        vocab[next_id] = A + B
        next_id += 1

        # 3.3 只更新包含该 pair 的 pre-token：撤销旧统计，写入合并后的新统计
        affected = list(pair_to_tokens.get(best_pair, ()))
        for token_tuple in affected:
            count = pre_token_counts.pop(token_tuple, 0)
            if count <= 0:
                continue
            _remove_pairs(token_tuple, count, pair_counts, pair_to_tokens, heap)
            new_tuple = merge_tuple(token_tuple, best_pair)
            pre_token_counts[new_tuple] += count
            _add_pairs(new_tuple, count, pair_counts, pair_to_tokens, heap)

        # 该 pair 的所有出现都已合并，确保它彻底从统计中移除
        pair_counts.pop(best_pair, None)
        pair_to_tokens.pop(best_pair, None)

    return vocab, merges
