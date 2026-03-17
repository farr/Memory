import pytest

from memory.hierarchical.data import _resolve_waveform_label


def test_resolve_waveform_label_auto_prefers_highest_priority_waveform():
    keys = ["C00:IMRPhenomXPHM", "C00:SEOBNRv5PHM", "C01:NRSur7dq4"]

    assert _resolve_waveform_label(keys, None) == "C01:NRSur7dq4"


def test_resolve_waveform_label_picks_highest_calibration_for_requested_waveform():
    keys = ["C00:NRSur7dq4", "C02:NRSur7dq4", "C01:NRSur7dq4"]

    assert _resolve_waveform_label(keys, "NRSur7dq4") == "C02:NRSur7dq4"


def test_resolve_waveform_label_accepts_exact_waveform_label():
    keys = ["C00:NRSur7dq4", "C01:NRSur7dq4"]

    assert _resolve_waveform_label(keys, "C00:NRSur7dq4") == "C00:NRSur7dq4"


def test_resolve_waveform_label_raises_for_missing_waveform():
    keys = ["C00:SEOBNRv5PHM", "C01:NRSur7dq4"]

    with pytest.raises(KeyError, match="Waveform 'IMRPhenomXPHM' not found"):
        _resolve_waveform_label(keys, "IMRPhenomXPHM")
