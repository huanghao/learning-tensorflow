import sys


if __name__ == '__main__':
    for line in sys.stdin:
        print line.replace('- ', '').replace('\r', '\n').replace('\n', '\n'*4)
