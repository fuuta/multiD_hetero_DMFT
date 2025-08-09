from dataclasses import dataclass

from serde import serialize

from ..utils.logging_utils import get_logger

logger = get_logger()


@serialize
@dataclass(frozen=True)
class Experimtnt:
    name: str
    description: str = ""
