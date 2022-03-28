import argparse
import src.dataspec.gen_data_spec as dataspec

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--dataspec', help='generates data specification', action="store_true")
args = parser.parse_args()

if args.dataspec:
    dataspec.run()