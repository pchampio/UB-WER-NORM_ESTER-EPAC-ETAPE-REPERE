import re
import glob
import json
import argparse
import subprocess

if __name__ ==  "__main__":

    desc = "Extract POS/normalize from stm output for UB-WER scoring"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        "--stm",
        required=True,
        help="Path to stm reference DIR"
    )
    parser.add_argument(
        "--stm-out",
        required=True,
        help="Path to stm reference DIR"
    )
    parser.add_argument(
        "--sb-pos-refs",
        required=True,
        help="Path to stm reference file"
    )
    args = parser.parse_args()

    file_pattern = '*.stm'
    
    file_list = glob.glob(args.stm + file_pattern)

    # Specify the output file

    output_file = args.stm_out.replace(".norm", ".concat")

    # Open the output file in write mode
    with open(output_file, 'w') as output:
        # Loop through each file in the list
        for file_path in file_list:
            if ".ne_e2." in file_path:
                continue
            if ".spk." in file_path:
                continue
            if ".ne." in file_path:
                continue
            # Open each file in read mode
            with open(file_path, 'r') as input_file:
                # Read the contents of the file and write them to the output file
                output.write(input_file.read())

    output_file = output_file.replace(".concat", "")

    process = subprocess.Popen(f"./normalize.v0.56 -if stm -d ./normalize.v2.1.dic  {output_file}.concat  -o {output_file}.norm -of tab ", shell=True)
    print(f"./normalize.v0.56 -if stm -d ./normalize.v2.1.dic  {output_file}.concat  -o {output_file}.norm -of tab ")
    process.wait()

    print(f"./normalize.v0.56 -if json -d ./normalize.v2.1.dic  {args.sb_pos_refs} -of json -o {args.sb_pos_refs}.norm -of json ")
    process = subprocess.Popen(f"./normalize.v0.56 -if json -d ./normalize.v2.1.dic  {args.sb_pos_refs} -of json -o {args.sb_pos_refs}.norm -of json ", shell=True)
    process.wait()


