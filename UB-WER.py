# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import deque
from enum import Enum
import subprocess
import re
import os
import pickle

from unidecode import unidecode
import argparse
import logging
import string
import json
import sys
import re

import spacy
nlp = spacy.load('fr_core_news_md')

UNIDECODE_RM_ACCENT = True

def uni(text):
    if UNIDECODE_RM_ACCENT:
        return unidecode(text).replace("'", "")
    return text

INS_COST_SCLITE = 3
DEL_COST_SCLITE = 3
SUB_COST_SCLITE = 4
INS_COST = 1
DEL_COST = 1
SUB_COST = 1

def cache_result(func, filename, *args, **kwargs):
    if os.path.exists(filename):
        with open(filename, 'rb') as file:
            cached_result = pickle.load(file)
        return cached_result
    else:
        result = func(*args, **kwargs)
        with open(filename, 'wb') as file:
            pickle.dump(result, file)
        return result

def make_multiple_words_prononciation_as_hesitation(list_wrd):
    out = []
    for i, t in enumerate(list_wrd):
        if i > 1 and list_wrd[i-1] == t:
            out.append(f"({t})")
        else:
            out.append(t)

    return out


class Code(Enum):
    match = 1
    substitution = 2
    insertion = 3
    deletion = 4
    hesitation = 10

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

def print_matrix(matrix):
    for row in matrix:
        print(" ".join(f"{cell:3}" for cell in row))
    print()

def _print_alignment(
    alignment, a, b, highlight_words=[], empty_symbol="<eps>", separator=" ; ", file=sys.stdout
):
    # First, get equal length text for all:
    a_padded = []
    b_padded = []
    ops_padded = []
    for op, i, j in alignment:  # i indexes a, j indexes b
        op_string = EDIT_SYMBOLS[op]
        a_string = str(a[i]) if i != -1 else empty_symbol
        b_string = str(b[j]) if j != -1 else empty_symbol
        # print(EDIT_SYMBOLS[op], a_string, b_string)
        # if op == Code.insertion:
        #     a_string = empty_symbol
        # NOTE: the padding does not actually compute printed length,
        # but hopefully we can assume that printed length is
        # at most the str len
        hi = False
        hi_b = False

        if unidecode(a_string).replace("'", "") in highlight_words or a_string in highlight_words:
            hi = True
            a_string = f"|{a_string}|"
        if unidecode(b_string).replace("'", "") in highlight_words or b_string in highlight_words:
            hi_b = True
            b_string = f"|{b_string}|"
        pad_length = max(len(op_string), len(a_string), len(b_string))
        if hi:
            a_string = f"{bcolors.UNDERLINE}{a_string}{bcolors.ENDC}"
        if hi_b:
            b_string = f"{bcolors.UNDERLINE}{b_string}{bcolors.ENDC}"
        a_padded.append(a_string.center(pad_length))
        b_padded.append(b_string.center(pad_length))
        ops_padded.append(op_string.center(pad_length))
    # Then print, in the order Ref, op, Hyp
    print(separator.join(a_padded), file=file)
    print(separator.join(ops_padded), file=file)
    print(separator.join(b_padded), file=file)
    print("================================================================================",file=file)


def _print_alignments_global_header(
    empty_symbol="<eps>", separator=" ; ", file=sys.stdout
):
    print("=" * 80, file=file)
    print("ALIGNMENTS", file=file)
    print("", file=file)
    print("Format:", file=file)
    print("<utterance-id>, WER DETAILS", file=file)
    # Print the format with the actual
    # print_alignment function, using artificial data:
    a = ["reference", "on", "the", "first", "line"]
    b = ["and", "hypothesis", "on", "the", "third"]
    alignment = [
        (Code.insertion, -1, 0),
        (Code.substitution, 0, 1),
        (Code.match, 1, 2),
        (Code.match, 2, 3),
        (Code.substitution, 3, 4),
        (Code.deletion, 4, -1),
    ]
    _print_alignment(
        alignment,
        a,
        b,
        file=file,
        empty_symbol=empty_symbol,
        separator=separator,
    )



def levenshtein_alignment(a, b, sclite_mode=True):
    def reverse_vector(vec):
        return vec[::-1]

    # Check inputs
    assert a is not None and b is not None
    assert isinstance(a, list) and isinstance(b, list)
    # assert all(x != eps_symbol for x in a)
    # assert all(x != eps_symbol for x in b)

    output = []

    ins_cost, del_cost, sub_cost = (INS_COST_SCLITE, DEL_COST_SCLITE, SUB_COST_SCLITE) if sclite_mode else (INS_COST, DEL_COST, SUB_COST)

    # Initialize the matrix
    M, N = len(a), len(b)
    e = [[0] * (N + 1) for _ in range(M + 1)]

    def cost(a, b, m, n):
        if a[m - 1][0] == "(" or a[m - 1][-1] == ")":
            sub_or_ok = e[m - 1][n - 1] + (0 if a[m - 1] == b[n - 1] else sub_cost)
            if a[m - 1].replace("(", "").replace(")", "")  == b[n - 1]:
                sub_or_ok = e[m - 1][n - 1] + 0
        else:
            sub_or_ok = e[m - 1][n - 1] + (0 if a[m - 1] == b[n - 1] else sub_cost)

        delete = e[m - 1][n] + del_cost
        insert = e[m][n - 1] + ins_cost

        return delete, insert, sub_or_ok, e


    for n in range(N + 1):
        e[0][n] = n * ins_cost

    for m in range(1, M + 1):
        e[m][0] = e[m - 1][0] + del_cost
        for n in range(1, N + 1):
            delete, insert, sub_or_ok, e = cost(a, b, m, n)
            e[m][n] = min(sub_or_ok, delete, insert)


    # Trace back to get the alignment
    m, n = M, N
    while m != 0 or n != 0:
        last_m, last_n = 0, 0
        if m == 0:
            last_m = m
            last_n = n - 1
        elif n == 0:
            last_m = m - 1
            last_n = n
        else:
            delete, insert, sub_or_ok, e = cost(a, b, m, n)
            if sub_or_ok < min(delete, insert):
                last_m, last_n = m - 1, n - 1
            else:
                if delete < insert:
                    last_m, last_n = m - 1, n
                else:
                    last_m, last_n = m, n - 1


        eps_symbol = "<eps>"
        a_sym = eps_symbol if last_m == m else a[last_m]
        b_sym = eps_symbol if last_n == n else b[last_n]

        # if len(b) < last_n:
        #     print(a_sym, b[last_n])

        # if "influence" in b and "l'" in b and eps_symbol == b_sym:
        #     print("20041026_1930_2000_RFI_ELDA-0162076-0162372", b, last_n, a_sym, b_sym, a[last_m-1])
        # if eps_symbol == b_sym and a[last_m-1].replace("(", "").replace(")", "") == a_sym:
        #     code = Code.hesitation
        if (a_sym[0] == "(" or a_sym[-1] == ")"):
            code = Code.hesitation
        elif uni(a_sym) == uni(b_sym):
            code = Code.match
        elif b_sym == eps_symbol:
            code = Code.deletion
        elif a_sym == eps_symbol:
            code = Code.insertion
        else:
            code = Code.substitution

        eps_symbol = -1

        a_sym_p = eps_symbol if last_m == m else last_m
        b_sym_p = eps_symbol if last_n == n else last_n
        output.append((code, a_sym_p, b_sym_p, a_sym, b_sym))
        m, n = last_m, last_n

    output = reverse_vector(output)
    return e[M][N], output

EDIT_SYMBOLS = {
    Code.match: "=",  # when tokens are equal
    Code.insertion: "I",
    Code.deletion: "D",
    Code.substitution: "S",
    Code.hesitation: "#",
}

class WordError(object):
    def __init__(self):
        self.errors = {
            Code.substitution: 0,
            Code.insertion: 0,
            Code.deletion: 0,
        }
        self.ref_words = 0

    def merge(self, object):
        object.errors[Code.substitution] += self.errors[Code.substitution]
        object.errors[Code.insertion] += self.errors[Code.insertion]
        object.errors[Code.deletion] += self.errors[Code.deletion]
        object.ref_words += self.ref_words

    def get_wer(self):
        if self.ref_words == 0:
            return 0
        errors = (
            self.errors[Code.substitution]
            + self.errors[Code.insertion]
            + self.errors[Code.deletion]
        )
        return 100.0 * errors / self.ref_words

    def get_sb_like_result_string(self):
        errors = (
            self.errors[Code.substitution]
            + self.errors[Code.insertion]
            + self.errors[Code.deletion]
        )
        return (f"%WER {self.get_wer():.2f} [ {errors} / {self.ref_words}, {self.errors[Code.insertion]} ins, {self.errors[Code.deletion]} del, {self.errors[Code.substitution]} sub ]")

    def get_result_string(self):
        return (
            f"error_rate={self.get_wer()}, "
            f"ref_words={self.ref_words}, "
            f"subs={self.errors[Code.substitution]}, "
            f"ins={self.errors[Code.insertion]}, "
            f"dels={self.errors[Code.deletion]}"
        )




def main(args):
    refs = {}
    pos_map_asr = {}
    with open(args.refs_asr, "r") as f:
        for line_asr in f:
            ary_asr = line_asr.strip().split("\t")
            if len(ary_asr) == 1:
                continue
            pos_map_asr[ary_asr[0]] = " ".join(ary_asr[1:])

            timestamp_parse = ary_asr[0].split("-")
            if int(timestamp_parse[-1]) - int(timestamp_parse[-2]) <= 50: # if sentance <= 50ms ignore scoring
                continue

    for index, refs_pos in enumerate(args.refs_pos.split(",")):
        with open(refs_pos, "r") as f:
            for line_pos in f:
                biasing_words_filter = []
                ary_pos = line_pos.strip().split("\t")
                if ary_pos[0] not in pos_map_asr:
                    raise ValueError(
                        f"Missing value seen in '{args.refs_pos}' ({ary_pos[0]}) but not in '{args.refs_asr}'"
                    )
                uttid, ref, biasing_words_pos = ary_pos[0], pos_map_asr[ary_pos[0]], json.loads(ary_pos[1])
                timestamp_parse = uttid.split("-")
                if int(timestamp_parse[-1]) - int(timestamp_parse[-2]) <= 50: # if sentance <= 50ms ignore scoring
                    continue

                biasing_words = []
                for t in args.pos_type.split(','):
                    if t.strip() in biasing_words_pos:
                        biasing_words += biasing_words_pos[t.strip()]
                biasing_words = list(map(lambda x: x.lower(), biasing_words))
                for words in biasing_words:
                    words_with_apostrophe = words.split(" ")
                    for part in words_with_apostrophe:
                        if "'" in part:
                            continue
                        else:
                            if uni(part).strip() != "" and part[0] != "(" and part[-1] != ")":
                                biasing_words_filter.append(uni(part).strip())


                if uttid not in refs:
                    refs[uttid] = {"text": ref, "biasing_words": biasing_words_filter}
                else:
                    refs[uttid]["biasing_words"] += biasing_words_filter

        for utt in refs.keys():
            refs[utt]["biasing_words"] = list(dict.fromkeys(refs[utt]["biasing_words"]))
            # print(index, utt, refs[utt]["biasing_words"])

        logger.info("Loaded %d reference utts %s", len(refs), refs_pos)

    BIASING_LIST = []
    if args.list_words:
        for index, refs_w_list in enumerate(args.list_words.split(",")):
            file, key = refs_w_list.split("|")
            if key in args.pos_type.split(','):
                with open(file) as f:
                    for word in f:
                        BIASING_LIST.append(list(map(lambda x: x.lemma_, nlp(word)))[0])



    hyps = {}
    with open(args.hyps, "r") as f:
        for line in f:
            ary = line.strip().split("\t")
            # May have empty hypo
            if len(ary) >= 2:
                uttid, hyp = ary[0], ary[1]
            else:
                uttid, hyp = ary[0], ""

            timestamp_parse = uttid.split("-")
            if int(timestamp_parse[-1]) - int(timestamp_parse[-2]) <= 50: # if sentance <= 50ms ignore scoring
                continue
            hyps[uttid] = hyp
    logger.info("Loaded %d hypothesis utts %s", len(hyps), args.hyps)

    if not args.lenient:
        for uttid in refs:
            if uttid in hyps:
                continue
            timestamp_parse = uttid.split("-")
            if int(timestamp_parse[-1]) - int(timestamp_parse[-2]) <= 50: # if sentance <= 50ms ignore scoring
                continue
            raise ValueError(
                f"{uttid} missing in hyps! Set `--lenient` flag to ignore this error."
            )

    # Calculate WER, U-WER, and B-WER
    wer_g = WordError()
    u_wer_g = WordError()
    b_wer_g = WordError()

    _print_alignments_global_header()

    refs = dict(sorted(refs.items(), key=lambda item: item[0]))
    for uttid in refs:
        if uttid not in hyps:
            continue
        ref_tokens = refs[uttid]["text"].lower().split()
        hyp_tokens = hyps[uttid].lower().split()

        input_string = refs[uttid]["text"]
        # Use regular expression to find all content within curly braces
        matches = re.finditer(r'\{([^}]+)\}', input_string)


        # Replace each matched expression with a unique word
        for match in matches:
            content_within_braces = match.group(1)
            words_within_braces = content_within_braces.split('/')
            input_string = input_string.replace(match.group(0), words_within_braces[0].strip(), 1)  # Replace only the first occurrence
            refs[uttid]["biasing_words"] += words_within_braces # TODO

            hyp_tokens = " ".join(hyp_tokens).replace(words_within_braces[1].strip(), words_within_braces[0].strip(), 1).split(" ")  # Replace only the first occurrence

        # Convert the result back to a list
        ref_tokens = input_string.lower().split()
        ref_tokens = make_multiple_words_prononciation_as_hesitation(ref_tokens)

        biasing_words = refs[uttid]["biasing_words"]
        result = levenshtein_alignment(ref_tokens, hyp_tokens)[1]
        nlp_ready = re.sub("\(%.*?\)", " ", " ".join(ref_tokens)).replace("(", "").replace(")", "").replace("'", "") 

        def lemma(nlp_ready):
            ref_tokens_nlp = list(map(lambda x: x.lemma_ , nlp(nlp_ready)) )
            return ref_tokens_nlp

        os.makedirs(".cache", exist_ok=True)
        ref_tokens_nlp = cache_result(lemma, f".cache/cache_lemma_{uttid}.pkl", nlp_ready)


        res = []
        rp_last = False
        for i, w in enumerate(ref_tokens_nlp):
            if rp_last:
                rp_last = False
                continue

            if "!" == w:
                continue
            # Consecutive hesitation handling after spacy
            if w == '        ' or w == '       ':
                res.append("  ")
                res.append("  ")
                res.append("  ")
                res.append("  ")
                continue
            if w == '      ' or w == '     ':
                res.append("  ")
                res.append("  ")
                res.append("  ")
                continue
            if w == '    ' or w == '   ':
                res.append("  ")
                res.append("  ")
                continue
            if i > 1 and w == '-':
                res[-1] += w+ref_tokens_nlp[i+1]
                rp_last = True
            else:
                res.append(w)

        ref_tokens_nlp_old = ref_tokens_nlp
        ref_tokens_nlp = res
        assert len(ref_tokens) == len(ref_tokens_nlp), f"{uttid} =\n({ref_tokens}) ({len(ref_tokens)}) \n({ref_tokens_nlp}) ({len(ref_tokens_nlp)}), {ref_tokens_nlp_old}"

        alignment = []
        wer = WordError()
        u_wer = WordError()
        b_wer = WordError()
        highlight_words_biasing_list = []
        for code, ref_idx, hyp_idx, ref, hyp in result:
            alignment.append((code, ref_idx, hyp_idx))
            if code == Code.match:
                wer.ref_words += 1
                if uni(ref_tokens[ref_idx]) in biasing_words or ref_tokens_nlp[ref_idx] in BIASING_LIST:
                    highlight_words_biasing_list.append(ref_tokens[ref_idx])
                    b_wer.ref_words += 1
                else:
                    u_wer.ref_words += 1
            elif code == Code.substitution:
                wer.ref_words += 1
                wer.errors[Code.substitution] += 1
                if uni(ref_tokens[ref_idx]) in biasing_words or ref_tokens_nlp[ref_idx] in BIASING_LIST:
                    highlight_words_biasing_list.append(ref_tokens[ref_idx])
                    b_wer.ref_words += 1
                    b_wer.errors[Code.substitution] += 1
                else:
                    u_wer.ref_words += 1
                    u_wer.errors[Code.substitution] += 1
            elif code == Code.deletion:
                wer.ref_words += 1
                wer.errors[Code.deletion] += 1
                if uni(ref_tokens[ref_idx]) in biasing_words or ref_tokens_nlp[ref_idx] in BIASING_LIST:
                    highlight_words_biasing_list.append(ref_tokens[ref_idx])
                    b_wer.ref_words += 1
                    b_wer.errors[Code.deletion] += 1
                else:
                    u_wer.ref_words += 1
                    u_wer.errors[Code.deletion] += 1
            elif code == Code.insertion:
                wer.errors[Code.insertion] += 1
                if uni(ref_tokens[ref_idx]) in biasing_words or ref_tokens_nlp[ref_idx] in BIASING_LIST:
                    highlight_words_biasing_list.append(ref_tokens[ref_idx])
                    b_wer.errors[Code.insertion] += 1
                else:
                    u_wer.errors[Code.insertion] += 1
        print(uttid + ",", wer.get_sb_like_result_string(), f"{bcolors.UNDERLINE}{b_wer.get_sb_like_result_string()}{bcolors.ENDC}")
        _print_alignment(alignment, ref_tokens, hyp_tokens, highlight_words=biasing_words + highlight_words_biasing_list, file=sys.stdout)
        wer.merge(wer_g)
        b_wer.merge(b_wer_g)
        u_wer.merge(u_wer_g)

    # Report results
    print(f"WER: {wer_g.get_result_string()}")
    print(f"U-WER: {u_wer_g.get_result_string()}")
    print(f"B-WER: {b_wer_g.get_result_string()}")


# Define digit mapping
romanNumeralMap = (('M', 1000),
                   ('CM', 900),
                   ('D', 500),
                   ('CD', 400),
                   ('C', 100),
                   ('XC', 90),
                   ('L', 50),
                   ('XL', 40),
                   ('X', 10),
                   ('IX', 9),
                   ('V', 5),
                   ('IV', 4),
                   ('I', 1))



# Define pattern to detect valid Roman numerals
romanNumeralPattern = re.compile("""
    ^                   # beginning of string
    M{0,4}              # thousands - 0 to 4 M's
    (CM|CD|D?C{0,3})    # hundreds - 900 (CM), 400 (CD), 0-300 (0 to 3 C's),
                        #            or 500-800 (D, followed by 0 to 3 C's)
    (XC|XL|L?X{0,3})    # tens - 90 (XC), 40 (XL), 0-30 (0 to 3 X's),
                        #        or 50-80 (L, followed by 0 to 3 X's)
    (IX|IV|V?I{0,3})    # ones - 9 (IX), 4 (IV), 0-3 (0 to 3 I's),
                        #        or 5-8 (V, followed by 0 to 3 I's)
    $                   # end of string
    """, re.VERBOSE)


def fromRoman(s):
    """convert Roman numeral to integer"""
    if not s:
        raise ValueError('Input can not be blank')

    # special case
    if s == 'N':
        return 0

    if not romanNumeralPattern.search(s):
        raise ValueError('Invalid Roman numeral: %s' % s)

    result = 0
    index = 0
    for numeral, integer in romanNumeralMap:
        while s[index:index + len(numeral)] == numeral:
            result += integer
            index += len(numeral)
    return result


def replace_roman_numerals(text, allow_lowercase=False, only_lowercase=False):
    """
    Replaces any roman numerals in 'text' with digits.
    Currently only looks for a roman numeral followed by a comma or period, then a space, then a digit.
    e.g. (Isa. Iv. 10) --> (Isa. 4:10)

    WARNING: we've seen e.g., "(v. 15)" used to mean "Verse 15". If run with allow_lowercase=True, this will
    be rewritten as "(5:15)".
    """
    if only_lowercase:
        regex = re.compile(r"((^|[{\[( ])[{\[( ]*)(m{0,4}(cm|cd|d?c{0,3})(xc|xl|l?x{0,3})(ix|iv|v?i{0,3}))(\. ?)(\d)?")
    else:
        flag = re.I if allow_lowercase else 0
        regex = re.compile(r"((^|[{\[( ])[{\[( ]*)(M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3}))($|[.,;\])}: ]+)(\d)?", flag)

    def replace_roman_numerals_in_match(m):
        s = m.group(3)
        s = s.upper()
        try:
            if s:
                if m.group(8):
                    return "{}{}:{}".format(m.group(1), fromRoman(s), m.group(8))
                else:
                    return "{}{}{}".format(m.group(1), fromRoman(s), m.group(7))
            else:
                return m.group(0)
        except:
            return m.group(0)

    return re.sub(regex, replace_roman_numerals_in_match, text)


if __name__ ==  "__main__":
    desc = "Compute WER, U-WER, and B-WER. Results are output to stdout."
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        "--refs-pos",
        required=True,
        help="Path to tab-separated reference file. First column is utterance ID. "
        "Last column is json of POS biasing words.",
    )
    parser.add_argument(
        "--list-words",
        help="Path to word file,type to add to POS"
    )
    parser.add_argument(
        "--refs-asr",
        required=True,
        help="Path to tab-separated reference file. First column is utterance ID. "
        "Second column is reference text.",
    )
    parser.add_argument(
        "--hyps",
        required=True,
        help="Path to tab-separated hypothesis file. First column is utterance ID. "
        "Second column is hypothesis text.",
    )
    parser.add_argument(
        "--pos-type",
        required=True,
        help="POS tag for the UB-WER (i.e., 'XFAMIL,PROPN')"
    )
    parser.add_argument(
        "--lenient",
        action="store_true",
        help="If set, hyps doesn't have to cover all of refs.",
    )
    args = parser.parse_args()
    with open(args.hyps, 'r') as input_file, open(f"{args.hyps}.tmp", 'w') as output_file:
        for line in input_file:
            output_file.write(replace_roman_numerals(line).replace("' ","'"))
    process = subprocess.Popen(f"./normalize.v0.56 -if tab -d ./normalize.v2.1.dic  {args.hyps}.tmp  -o {args.hyps}.norm", shell=True)
    process.wait()
    args.hyps = f"{args.hyps}.norm"
    main(args)
