import sys

from google.protobuf.text_format import Merge
from tensorflow.core.framework.graph_pb2 import GraphDef

gdef = GraphDef()
Merge(sys.stdin.read(), gdef)

print('digraph g {')
for node in gdef.node:
    for name in node.input:
        print('"%s" -> "%s";' % (name, node.name))
print('}')
