"""
grep import *.proto | python proto_dot.py | dot -Tpng >proto.png
"""
import os
import sys


def find_proto(top):
    names = {}
    for dirpath, dirnames, filenames in os.walk(top):
        for filename in filenames:
            filename = os.path.join(dirpath, filename)
            if filename.endswith('.proto'):
                base = os.path.basename(filename)
                name, _ = os.path.splitext(base)
                if name not in names:
                    names[name] = filename
                else:
                    oldfilename = names.pop(name)
                    olddir = os.path.basename(os.path.dirname(oldfilename))
                    newname1 = '.'.join([olddir, name])
                    names[newname1] = oldfilename

                    newdir = os.path.basename(os.path.dirname(filename))
                    newname2 = '.'.join([newdir, name])
                    names[newname2] = filename
    return names


def extract_import(filename, names):
    im = []
    with open(filename) as f:
        for line in f:
            if line.startswith('import "'):
                mod = line.split('"')[1]
                mod, _ = os.path.splitext(mod)
                mod = mod.split('/')
                while mod:
                    n = '.'.join(mod)
                    if n in names:
                        im.append(n)
                    mod.pop(0)
    return im


top = '/Users/huanghao/workspace/google/tensorflow/tensorflow'

names = find_proto(top)

print 'digraph g {'
for name, path in sorted(names.items()):
    deps = extract_import(path, names)
    for dep in deps:
        print '  "%s" -> "%s";' % (name, dep)
print '}'
