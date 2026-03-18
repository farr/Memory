import h5py
import numpy as np
import pytest

from memory.hierarchical.data import _resolve_waveform_label, load_memory_data


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


def test_load_memory_data_skips_events_missing_requested_waveform(tmp_path):
    event_file = tmp_path / "GW190814_211039_posterior.h5"
    event_file.write_text("")

    memory_dir = tmp_path / "memory"
    event_dir = memory_dir / "GW190814_211039"
    event_dir.mkdir(parents=True)

    with h5py.File(event_dir / "memory_results.h5", "w") as f:
        group = f.create_group("C00:SEOBNRv5PHM")
        values = np.array([1.0, 2.0])
        group.create_dataset("A_sample", data=values)
        group.create_dataset("A_hat", data=values)
        group.create_dataset("A_sigma", data=values)
        group.create_dataset("log_weight", data=values)

    memory_data = load_memory_data(
        [str(event_file)],
        str(memory_dir),
        "NRSur7dq4",
    )

    assert memory_data == []


def test_load_memory_data_keeps_highest_calibration_for_requested_waveform(tmp_path):
    event_file = tmp_path / "GW190814_211039_posterior.h5"
    event_file.write_text("")

    memory_dir = tmp_path / "memory"
    event_dir = memory_dir / "GW190814_211039"
    event_dir.mkdir(parents=True)

    with h5py.File(event_dir / "memory_results.h5", "w") as f:
        for label, value in [("C00:NRSur7dq4", 1.0), ("C02:NRSur7dq4", 2.0)]:
            group = f.create_group(label)
            values = np.array([value])
            group.create_dataset("A_sample", data=values)
            group.create_dataset("A_hat", data=values)
            group.create_dataset("A_sigma", data=values)
            group.create_dataset("log_weight", data=values)

    memory_data = load_memory_data(
        [str(event_file)],
        str(memory_dir),
        "NRSur7dq4",
    )

    assert [item["waveform_label"] for item in memory_data] == ["C02:NRSur7dq4"]
