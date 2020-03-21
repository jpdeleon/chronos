# -*- coding: utf-8 -*-

# Import modules
import pytest

# Import from package
from chronos.star import Star


def test_cluster():
    s = Star(toiid=837)
    assert s.ticid is not None
    assert s.toiid is not None
