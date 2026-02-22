from enum import Enum


class ArmSide(Enum):
    LEFT = "left"
    RIGHT = "right"


class CameraPerspective(Enum):
    TOP = "top"
    LEFT = "left"
    RIGHT = "right"


# All right handed
class FrameConvention(Enum):
    # FLU, X forward, Y left, Z up
    WORLD_FLU = "world_flu"
    # RDF, X right, Y down, Z forward
    GRIPPER_RDF = "gripper_rdf"
    # URF, X up, Y right, Z forward
    GRIPPER_URF = "gripper_urf"
