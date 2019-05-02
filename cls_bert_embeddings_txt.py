#!/usr/bin/env python3
import argparse
import re
import sys
import zipfile

import numpy as np

import bert_wrapper

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_txt", type=str, help="Input TXT file")
    parser.add_argument("output_txt", type=str, help="Output TXT file")
    parser.add_argument("--batch_size", default=16, type=int, help="Batch size")
    parser.add_argument("--casing", default=bert_wrapper.BertWrapper.CASING_UNCASED, help="Bert model casing")
    parser.add_argument("--language", default=bert_wrapper.BertWrapper.LANGUAGE_MULTILINGUAL, help="Bert model language")
    parser.add_argument("--layer_indices", default="-1,-2,-3,-4", type=str, help="Bert model layers to average")
    parser.add_argument("--size", default=bert_wrapper.BertWrapper.SIZE_BASE, help="Bert model size")
    parser.add_argument("--threads", default=4, type=int, help="Threads to use")
    args = parser.parse_args()
    args.layer_indices = list(map(int, args.layer_indices.split(",")))

    # Load TXT file
    sentences = []
    with open(args.input_txt, mode="r", encoding="utf-8") as input_txt:
        for line in input_txt:
            sentences.append(line.split())
    print("Loaded TXT file with {} sentences and {} words.".format(len(sentences), sum(map(len, sentences))), file=sys.stderr)

    bert = bert_wrapper.BertWrapper(language=args.language, size=args.size, casing=args.casing, layer_indices=args.layer_indices,
                                    with_cls=True, threads=args.threads, batch_size=args.batch_size)
    with open(args.output_txt, mode="w") as output_txt:
        for i, embeddings in enumerate(bert.bert_embeddings(sentences)):
            if (i + 1) % 100 == 0: print("Processed {}/{} sentences.".format(i + 1, len(sentences)), file=sys.stderr)
            print(embeddings.tolist()[0], file=output_txt)
    print("Done, all embeddings saved.", file=sys.stderr)
