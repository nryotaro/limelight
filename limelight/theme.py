"""
"""
import enum
from typing import List
from dataclasses import dataclass


class Theme(enum.Enum):
    """Represent the labels."""

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

    def get_theme_name(self) -> str:
        """Get the name of the directory.

        Returns
        -------
        str
            directory name.

        """
        if self == self.COMP_OS_MS_WINDOWS_MISC:
            return 'comp.os.ms-windows.misc'
        return self.name.replace('_', '.').lower()

    @classmethod
    def get_themename_list(cls) -> List[str]:
        """Return the names of the themes."""
        return [theme.get_theme_name() for theme in cls]

    def is_same(self, theme_name: str) -> bool:
        """Return `True` if `them_name` is the name of the directory."""
        return self.get_theme_name() == theme_name

    @classmethod
    def create(cls, theme: str):
        """Create :py:class:`Theme` from a `str`."""
        found = [item for item in Theme if item.is_same(theme)]
        if len(found) == 1:
            return found[0]
        raise ValueError(f'{theme} is not a theme name')


@dataclass
class Themes:
    """A collection of :py:class:`Themes`."""

    themes: List[Theme]

    def get_index(self) -> List[int]:
        """
        """
        raise NotImplementedError
