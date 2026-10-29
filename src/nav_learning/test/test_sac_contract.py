from dataclasses import replace
import math
import numpy as np
import pytest
from nav_learning.sac_math import Settings, scan_sectors, build_observation, steering_velocity, reward_for_transition, InvalidScanError
from nav_learning.tasks import sample_task
from nav_learning.arena import navigation_path_exists, _sample_random_episode


def test_body_frame_goal_and_rotation_equivariance():
    lidar = np.ones(20)
    a = build_observation(lidar, 1, 2, 0, (4, 2))
    b = build_observation(lidar, -2, 1, math.pi/2, (-2, 4))
    np.testing.assert_allclose(a, b, atol=1e-7)
    assert a.shape == (22,) and a.dtype == np.float32
    assert a[-2] > 0 and abs(a[-1]) < 1e-7


def test_lidar_min_pool_preserves_narrow_front_return():
    angles = np.linspace(-math.pi, math.pi, 360)
    rays = np.full(360, 5.)
    rays[np.argmin(abs(angles))] = .21
    bins, nearest, front = scan_sectors(rays, -math.pi, 2*math.pi/359)
    assert nearest == pytest.approx(.21)
    assert front == pytest.approx(.21)
    assert bins[10] == pytest.approx(.21/8)


def test_lidar_invalid_data_and_coverage():
    rays = np.full(360, np.nan)
    with pytest.raises(InvalidScanError):
        scan_sectors(rays, -math.pi, 2*math.pi/359)
    with pytest.raises(InvalidScanError):
        scan_sectors(np.ones(360), 0, math.pi/359)
    sectors, nearest, _ = scan_sectors(np.full(360, np.inf), -math.pi, 2*math.pi/359)
    assert nearest == 8 and np.all(sectors == 1)


def test_one_action_constant_speed_and_no_speed_plateau():
    settings = Settings()
    for a in (-1, 0, 1):
        v, w = steering_velocity([a], settings)
        assert v == .22 and w == pytest.approx(a*1.2)
    with pytest.raises(ValueError):
        steering_velocity([0, 0], settings)
    with pytest.raises(ValueError):
        Settings(fixed_speed_mps=3)


def test_reward_terminals_truncation_and_passage_margin():
    settings = Settings()
    timeout, parts = reward_for_transition(8, 8, .325, False, "timeout", settings)
    running, _ = reward_for_transition(8, 8, .325, False, "running", settings)
    assert timeout == running and parts["near"] == 0
    _, terminal = reward_for_transition(8, 7.98, .215, False, "collision", settings)
    assert terminal["potential"] == 8 and terminal["collision"] == -20
    _, face = reward_for_transition(8, 7.98, .4, False, "running", settings, front=.4)
    assert face["front"] < 0 and face["near"] == 0
    _, gap = reward_for_transition(8, 7.98, .325, False, "running", settings, front=.9)
    assert gap["front"] == 0 and gap["near"] == 0


def test_potential_shaping_telescopes_over_true_terminal():
    settings = Settings()
    ds = [8, 7.9, 8.1, 5, 0]
    terms = []
    for i in range(4):
        _, parts = reward_for_transition(ds[i], ds[i+1], .5, False,
                                         "success" if i == 3 else "running", settings)
        terms.append(parts["potential"])
    assert sum(settings.gamma**i * x for i, x in enumerate(terms)) == pytest.approx(8)


@pytest.mark.parametrize("stage", range(5))
def test_seeded_tasks_are_reachable_and_not_fixed(stage):
    tasks = [sample_task(520000 + stage*10000+i, stage, training=False) for i in range(8)]
    assert len({t.goal for t in tasks}) > 1
    for task in tasks:
        again = sample_task(task.seed, stage, training=False)
        assert task == again
        assert navigation_path_exists(task.start[:2], task.goal, task.layout)
        assert len(task.layout) <= 15
        if task.kind == "passage":
            assert .65 <= task.gap_m <= 1.3


def test_standard_layouts_preserve_original_final_stage():
    import hashlib
    import json
    # Digests come from the separately uploaded evaluation_layouts(2).json,
    # not from this copy of the geometry sampler.
    expected = {140000: "a6e1284e387cfed3395e406e6eca5dc7ab11f6162ff53b8cd5be7b765cf804d5",
                140002: "eb644bac7f64b5f68c9f1535c90aac83dbe4eab3f66ed172a26624aa32a605c4",
                140049: "6031ec8a8bfd4319b6929adb198a4a2f3c8b63a23a54cc6e8af9ff28f3fc0d3d"}
    for seed in expected:
        a = sample_task(seed, 4, benchmark=True)
        fields = {"layout": a.layout, "start": a.start, "goal": a.goal}
        digest = hashlib.sha256(json.dumps(fields,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        assert digest == expected[seed]
        assert a.max_steps == 1600
