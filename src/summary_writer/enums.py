from typing import ClassVar
from ai_common import NodeBase
from pydantic import ConfigDict

class Node(NodeBase):
    model_config = ConfigDict(frozen=True)

    # Class attributes
    REVIEWER: ClassVar[str] = 'reviewer'
    WRITER: ClassVar[str] = 'writer'
    CITATIONS_MANAGER: ClassVar[str] = 'citations_manager'
