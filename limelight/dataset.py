"""
"""
import os
from enum import Enum
from dataclasses import dataclass
import re


class Theme(Enum):
    """
    """

    TALK_POLITICS_MIDEAST = 0
    REC_AUTOS = 1
    COMP_SYS_MAC_HARDWARE = 2
    ALT_ATHEISM = 3
    REC_SPORT_BASEBALL = 4
    COMP_OS_MS_WINDOWS_MISC = 5
    REC_SPORT_HOCKEY = 6
    SCI_CRYPT = 7
    SCI_MED = 8
    TALK_POLITICS_MISC = 9
    REC_MOTORCYCLES = 10
    COMP_WINDOWS_X = 11
    COMP_GRAPHICS = 12
    COMP_SYS_IBM_PC_HARDWARE = 13
    SCI_ELECTRONICS = 14
    TALK_POLITICS_GUNS = 15
    SCI_SPACE = 16
    SOC_RELIGION_CHRISTIAN = 17
    MISC_FORSALE = 18
    TALK_RELIGION_MISC = 19

    @classmethod
    def create(cls, theme: str):
        """
        """
        """
        ['talk.politics.mideast',
        'rec.autos',
 'comp.sys.mac.hardware',
 'alt.atheism',
 'rec.sport.baseball',
 'comp.os.ms-windows.misc',
 'rec.sport.hockey',
 'sci.crypt',
 'sci.med',
 'talk.politics.misc',
 'rec.motorcycles',
 'comp.windows.x',
 'comp.graphics',
 'comp.sys.ibm.pc.hardware',
 'sci.electronics',
 'talk.politics.guns',
 'sci.space',
 'soc.religion.christian',
 'misc.forsale',
 'talk.religion.misc'].index(theme)
        """

class Splitter:
    """
    """

    def __init__(self, directory: str, seed=None):
        """

        Parameters
        ----------
        directory: str

        seed : int or None
            used by a random number generator.

        """

    def split(self):
        """
        """
        os.listdir('./hoge')[0].startswith('.')
