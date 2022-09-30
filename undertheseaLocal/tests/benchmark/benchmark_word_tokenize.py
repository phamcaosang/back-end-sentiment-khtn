import cProfile
import pstats
from os.path import join

from underthesea.file_utils import DATASETS_FOLDER
from underthesea.pipeline.word_tokenize import tokenize

# ======================================================
# PERFORMANCE
# ======================================================
# DATA
# 1000 sentences, 50k tokens
# ======================================================
# v1.1.3     :  3.771s
# Nightly    :  1.545s !!! >.<
# Expected 1 :  0.025s (speed:  2M tokens/s)
# Expected 2 :  0.002s (speed: 20M tokens/s)
# ======================================================

total_sentence = 0
total_tokens = 0


def get_sentences():
    global total_sentence
    global total_tokens
    sentences = []
    with open(join(DATASETS_FOLDER, "LTA", "VNESEScorpus.txt")) as f:
        for i, line in enumerate(f):
            sentences.append(line)
            tokens = tokenize(line)
            total_tokens += len(tokens)
            total_sentence += 1
            if i > 1000:
                break

    print(f"Load {total_sentence} sentences, {total_tokens} tokens")
    print("=========================================")
    return sentences


sentences = get_sentences()


def word_tokenize_old():
    from underthesea import word_tokenize
    for s in sentences:
        word_tokenize(s)


def word_tokenize_new():
    from underthesea.pipeline.word_tokenize.nightly import word_tokenize as word_tokenize_nightly
    for s in sentences:
        word_tokenize_nightly(s)


profiler = cProfile.Profile()
profiler.enable()
word_tokenize_old()
profiler.disable()
stats = pstats.Stats(profiler).sort_stats('tottime')
stats.print_stats()

run_time = stats.total_tt
print('Current Speed')
sentences_per_sec = total_sentence / run_time
tokens_per_sec = total_tokens / run_time
print(f'{sentences_per_sec:06.2f} sentences/sec')
print(f'{tokens_per_sec:06.2f} tokens/sec')
