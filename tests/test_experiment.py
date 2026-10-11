import json
from dataclasses import replace

import numpy as np
import pytest

from simengine.experiment import config_hash, paired_difference, run_experiment
from simengine.station import MMcConfig, mmc_model

BASE = MMcConfig(arrival_rate=2.4, service_rate=1.0, servers=3, horizon=3_000, warmup=300)


def test_same_seed_reproduces_every_kpi():
    a = run_experiment(mmc_model, BASE, replications=3, base_seed=99)
    b = run_experiment(mmc_model, BASE, replications=3, base_seed=99)
    assert a.replications == b.replications


def test_config_hash_is_stable_and_sensitive():
    assert config_hash(BASE) == config_hash(replace(BASE))
    assert config_hash(BASE) != config_hash(replace(BASE, servers=4))


def test_single_run_is_refused():
    with pytest.raises(ValueError):
        run_experiment(mmc_model, BASE, replications=1, base_seed=0)


def test_record_round_trips_to_disk(tmp_path):
    record = run_experiment(mmc_model, BASE, replications=2, base_seed=3)
    saved = json.loads(record.save(tmp_path).read_text())
    assert saved["config_hash"] == record.config_hash
    assert saved["base_seed"] == 3
    assert saved["engine_version"] == record.engine_version
    assert len(saved["replications"]) == 2


def test_common_random_numbers_tighten_scenario_comparison():
    # Three lanes with a 5% faster screening process. With a shared seed every passenger arrives at
    # the same moment and needs proportionally the same service in both scenarios, so the
    # difference in mean wait varies far less than between independently seeded runs.
    reps = 15
    faster = replace(BASE, service_rate=1.05)
    base = run_experiment(mmc_model, BASE, reps, base_seed=10)
    crn = run_experiment(mmc_model, faster, reps, base_seed=10)
    indep = run_experiment(mmc_model, faster, reps, base_seed=11)

    crn_sd = np.std(crn.kpi("Wq") - base.kpi("Wq"), ddof=1)
    indep_sd = np.std(indep.kpi("Wq") - base.kpi("Wq"), ddof=1)
    assert crn_sd < 0.5 * indep_sd
    assert paired_difference(base, crn, "Wq").high < 0  # CRN resolves the improvement


def test_pairing_requires_shared_seed():
    a = run_experiment(mmc_model, BASE, 2, base_seed=1)
    b = run_experiment(mmc_model, BASE, 2, base_seed=2)
    with pytest.raises(ValueError):
        paired_difference(a, b, "Wq")
