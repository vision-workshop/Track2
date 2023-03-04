import argparse
import subprocess

def parse_args():
    parser = argparse.ArgumentParser(description='Block 2 Datasets')
    parser.add_argument('--datasets', type=str, nargs='+', required=True,
                        help='datasets to train and test on')
    return parser.parse_args()

def main():
    args = parse_args()

    for dataset in args.datasets:
        print(f"Running Block 2 training for dataset: {dataset}")
        train_cmd = f"python tools/train.py config.py {dataset} --seed 0 --deterministic --gpu-ids 0"
        p = subprocess.Popen(train_cmd, stdout=subprocess.PIPE, shell=True)
        p.communicate()

        #test result
        print(f"Running Block 2 testing for dataset: {dataset}")
        test_cmd = f"python tools/test.py config.py {dataset} \
            work_dirs/{dataset}/latest.pth --format-only --eval-options jsonfile_prefix=./results/{dataset}"
        q1 = subprocess.Popen(test_cmd, stdout=subprocess.PIPE, shell=True)
        q1.communicate()

    print("Done")


if __name__ == "__main__":
    main()
