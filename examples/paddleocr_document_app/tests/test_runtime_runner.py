"""Developer runner tests, separate from submitted application contract files."""
from pathlib import Path
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from run_upgrade_demo import runner_parameters


def test_controlled_configuration_runner_parameter_is_bounded_and_version_specific():
    assert runner_parameters('2.9.1',None)=={}
    assert runner_parameters('2.9.1',0.8)=={'drop_score':0.8}
    for version,value in [('3.0.0',0.8),('2.9.1',True),('2.9.1',float('nan')),
                          ('2.9.1',float('inf')),('2.9.1',-0.1),('2.9.1',1.1),('2.9.1',10**500)]:
        with pytest.raises(ValueError):runner_parameters(version,value)
