from typing import ClassVar
from ai_common import NodeBase

"""
# Extended nodes (with a deliberate duplicate key to test conflict resolution)
extra_node_members = {
    'REVIEWER': 'reviewer',
    'WRITER': 'writer',
    'QUERY_WRITER': 'should_be_ignored'
}

# Build the final Node enum
Node = build_node_enum(base_enum=NodeBase, extra_nodes=extra_node_members)
"""

class Node(NodeBase):
    REVIEWER: ClassVar[str] = 'reviewer'
    WRITER: ClassVar[str] = 'writer'

    class Config:
        allow_mutation = False
