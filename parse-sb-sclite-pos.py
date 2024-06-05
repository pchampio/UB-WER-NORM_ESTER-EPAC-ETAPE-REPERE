import re
import json
import argparse
import subprocess

def parse_integer_to_float(input_str):
    # Extract the integer part (before the last three digits)
    integer_part = input_str[:-2]
    
    # Extract the decimal part (last three digits)
    decimal_part = input_str[-2:]
    
    # Combine the integer and decimal parts with a dot in between
    result = f"{integer_part}.{int(decimal_part)+2}".lstrip("0")
    
    return result

def parse_sclite_output(sclite_output):
    utterances = []
    
    # Define regular expressions
    utterance_pattern = re.compile(r'(.+), %WER (\d+\.\d+) \[ (\d+) \/ (\d+), (\d+) ins, (\d+) del, (\d+) sub \]\n(.+)', re.DOTALL)
    
    # Split the input by '================================================================================'
    blocks = sclite_output.split("================================================================================")
    
    for block in blocks:
        match = utterance_pattern.match(block)
        if match:
            utterance_id = match.group(1).strip()
            wer = float(match.group(2))
            details = {
                'ins': int(match.group(5)),
                'del': int(match.group(6)),
                'sub': int(match.group(7))
            }
            hypothesis = [m.strip() for m in match.group(8).split('\n')[5].split(";")]
            reference = [m.strip() for m in match.group(8).split('\n')[1].split(";")]

            utterances.append({
                'utterance_id': utterance_id,
                'wer': wer,
                'details': details,
                'reference': reference,
                'hypothesis': hypothesis
            })
    
    return utterances

if __name__ ==  "__main__":

    desc = "Extract POS from sclite like output for UB-WER scoring"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        "--sclite",
        required=True,
        help="Path to Sclite file"
    )
    parser.add_argument(
        "--hyps",
        required=True,
        help="Path to tab-separated hypothesis file. To generate"
    )
    parser.add_argument(
        "--refs",
        required=True,
        help="Path to tab-separated reference file. To generate"
    )
    parser.add_argument(
        "--hyps-only",
        required=False,
        help="Only run hyps file generation"
    )
    args = parser.parse_args()


    if args.hyps_only == None:
        args.hyps_only = "false"

    if args.hyps_only.lower() != "true":
        from flair.data import Sentence
        from flair.models import SequenceTagger

        # Load the model
        model = SequenceTagger.load("qanastek/pos-french")

    file_path = args.sclite
    with open(file_path, 'r') as file:
        sclite_output = "\n".join(file.readlines())
    result = parse_sclite_output(sclite_output)


    file_path = args.refs
    file_path2 = args.hyps
    if args.hyps_only.lower() == "true":
        file_path = "/tmp/rm"

    # Open the file in write mode
    with open(file_path, 'w') as filerr, open(file_path2, 'w')  as filerh, open(file_path2+".ctm", 'w')  as ctm:

        # Accessing the parsed data
        for utterance in result:
            # print(f"Utterance ID: {utterance['utterance_id']}")
            # print(f"WER: {utterance['wer']}")
            # print(f"Details: {utterance['details']}")
            # print(f"Reference: {utterance['reference']}")
            # print(f"Hypothesis: {utterance['hypothesis']}")
            # print("")

            if args.hyps_only.lower() != "true":
                sentence = Sentence(" ".join(utterance['reference']).replace("'", "'").replace("<eps>", ""))
                model.predict(sentence)

                # Print predicted pos tags
                print(sentence.to_tagged_string())
                utterance["POS"] = {}
                for l in sentence.labels:
                    # print(l.value, l.shortstring.split("\"")[1])
                    if str(l.value) not in utterance["POS"]:
                        utterance["POS"][str(l.value)] = set()
                    utterance["POS"][str(l.value)].add(l.shortstring.split("\"")[1])

                sentence = Sentence(" ".join(utterance['reference']).replace("'", "'").replace("<eps>", "").replace("'", "' "))
                model.predict(sentence)

                for l in sentence.labels:
                    if str(l.value) not in utterance["POS"]:
                        utterance["POS"][str(l.value)] = set()

                    utterance["POS"][str(l.value)].add(l.shortstring.split("\"")[1])


                for k, v in utterance["POS"].items():
                    utterance["POS"][k] = list(map(lambda x: f" {x} ", v))

            # Write the string representation of the set to the file
            if args.hyps_only.lower() != "true":
                filerr.write(utterance['utterance_id'] + "\t" + " ".join(utterance['reference']).replace("<eps>","") + "\t " + json.dumps(utterance["POS"],  ensure_ascii=False) +"\n")
            filerh.write(utterance['utterance_id'] + "\t" + " ".join(utterance['hypothesis']).replace("<eps>","") +"\n")
            for i, w in enumerate(utterance['hypothesis']):
                if w == "<eps>":
                    continue
                ctm.write(utterance['utterance_id'].split("-")[0] + " 1 " + str(parse_integer_to_float(utterance['utterance_id'].split("-")[2])) + " " + str(0.01*i) + " " + w + "\n")
