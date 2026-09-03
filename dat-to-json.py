import argparse
import json
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pyhs3-nlls", type=str, help="The pyhs3 nll .dat file to convert to JSON")
    parser.add_argument("--qf-nlls", type=str, help="The qf nll .dat file to convert to JSON")
    parser.add_argument("--out-file", default="data.json", type=str, help="The output JSON file")
    args = parser.parse_args()
    
    with open(args.pyhs3_nlls, "r") as f:
        pyhs3_nlls = f.readlines()
    with open(args.qf_nlls, "r") as f:
        qf_nlls = f.readlines()

    with open(args.pyhs3_nlls, "r") as f:
        pyhs3_lines = [line.split() for line in f if line.strip()]
        pyhs3_nlls = [float(line[1]) for line in pyhs3_lines[1:]]
        mus1 = [float(line[0]) for line in pyhs3_lines[1:]]
    with open(args.qf_nlls, "r") as f:
        qf_lines = [line.split() for line in f if line.strip()]
        qf_nlls = [float(line[1]) for line in qf_lines[1:]]
        mus2 = [float(line[0]) for line in qf_lines[1:]]

    assert len(pyhs3_lines) == len(qf_lines)

    data = {
        "workspace": args.pyhs3_nlls.split("/")[-1].split(".")[0],
        "scan": "",
        "mus": mus1,
        "diffs": [qf_nlls[i] - pyhs3_nlls[i] for i in range(len(mus1))],
        "mean_offset": None,
        "max_abs_resid": None,
        "qf_nlls": qf_nlls,
        "pyhs3_nlls": pyhs3_nlls,
        "offset (N*ln(C))": None,
    }

    outfile = args.out_file
    with open(outfile, "w") as f:
        json.dump(data, f, indent=4)

    return



if __name__ == "__main__":
    main()