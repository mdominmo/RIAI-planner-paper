from enum import Enum

class AssignationMethods(Enum):
    RANDOM = 0
    RRT_STAR_HUNGARIAN = 1
    ONLY_RRT_STAR = 2
    BY_EUCLIDEAN_CRITERIA = 3

