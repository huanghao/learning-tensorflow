import sys
import argparse
import subprocess


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('-c', '--copy', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()

    input_ = sys.stdin.read()
    text = input_.replace('- ', '').replace('\r', '\n'*2)

    if args.copy:
        p = subprocess.Popen(['pbcopy'], stdin=subprocess.PIPE, close_fds=True)
        p.communicate(input=text)
    else:
        print text


if __name__ == '__main__':
    main()
