"""The scripted kitchen behind `roomtone run --sim`."""

import numpy as np
import pytest

from roomtone.kitchen import DEFAULT_SCRIPT, KitchenScript, format_script, parse_script

SR = 16_000


def rms_db(x: np.ndarray) -> float:
    return float(20 * np.log10(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2))))


def test_parse_and_format_script():
    assert parse_script("empty:150, Cooking:180,dinner:2.5") == [
        (150.0, "empty"), (180.0, "cooking"), (2.5, "dinner")]
    assert parse_script(format_script(DEFAULT_SCRIPT)) == DEFAULT_SCRIPT
    for bad in ["", "empty", "disco:10", "empty:0", "empty:soon", "empty:10,,party:5"]:
        with pytest.raises(ValueError):
            parse_script(bad)


def test_phase_at_follows_the_script_and_repeats():
    script = KitchenScript([(10, "empty"), (5, "party")], SR)
    assert [script.phase_at(t) for t in (0, 9.9, 10, 14.9, 15, 26)] == [
        "empty", "empty", "party", "party", "empty", "party"]


def test_streams_the_same_whatever_the_request_size():
    phases = [(3, "cooking"), (6, "dinner"), (2, "party")]
    whole = KitchenScript(phases, SR, seed=3).next(12 * SR)
    pieces = KitchenScript(phases, SR, seed=3)
    sizes = np.random.default_rng(0).integers(1, 9000, 400)
    parts, total = [], 0
    for n in sizes:
        n = int(min(n, 12 * SR - total))
        parts.append(pieces.next(n))
        total += n
        if total == 12 * SR:
            break
    np.testing.assert_array_equal(np.concatenate(parts), whole)


def test_levels_per_mode():
    script = KitchenScript([(20, "empty"), (20, "cooking"), (20, "dinner"), (20, "party")],
                           SR, noise_db=-60.0)
    x = script.next(80 * SR)
    levels = [rms_db(x[(20 * i + 1) * SR:(20 * i + 19) * SR]) for i in range(4)]
    assert levels[0] == pytest.approx(-60.0, abs=0.5)
    assert levels[1] == pytest.approx(-32.0, abs=2.0)    # clatter plus a faint hum
    assert levels[2] == pytest.approx(-27.5, abs=1.5)    # talk plus a little clatter
    assert levels[3] == pytest.approx(-16.5, abs=1.5)    # crowd at -18 plus kick at -22


def test_party_kick_is_a_steady_120_bpm_at_minus_22():
    script = KitchenScript([(4, "party")], SR)
    kick = script._fixed("party", 0, 4 * SR, 0, 4 * SR)
    beat = SR // 2
    steady = kick[2 * beat:]
    np.testing.assert_allclose(steady[beat:], steady[:-beat], atol=1e-9)   # repeats every 0.5 s
    assert rms_db(steady[:4 * beat]) == pytest.approx(-22.0, abs=0.1)
    assert kick[0] == 0.0 and np.argmax(np.abs(kick[:beat])) == pytest.approx(SR / 240, abs=2)


def test_chunk_and_phase_boundaries_do_not_click():
    # talk and kick only (no clatter), so every big sample-to-sample jump would be a seam
    script = KitchenScript([(7, "party"), (3, "empty"), (8, "party")], SR, seed=1)
    x = script.next(18 * SR).astype(np.float64)
    step = np.abs(np.diff(x))
    seams = [5, 7, 10, 15]
    near = max(step[s * SR - 40:s * SR + 40].max() for s in seams)
    inside = step[int(0.5 * SR):int(4.5 * SR)].max()
    assert near <= inside
    assert rms_db(x[int(7.1 * SR):int(9.9 * SR)]) == pytest.approx(-60.0, abs=0.5)
