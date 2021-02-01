import sys


def convert_file_format():
    """Converts file format of viterbi.sh system to word/tag pairs."""
    for line in sys.stdin:
        symbols, tags = line.strip().split('=>')
        tags = tags.split()[:-1]
        tags = tags[1:]
        tags = map(lambda x: x.split('_')[-1], tags)
        symbols = symbols.split()
        print(' '.join(map('/'.join, zip(symbols, tags))))


if __name__ == "__main__":
    convert_file_format()
