from dataclasses import replace
from pathlib import Path
import json
import random
import numpy as np
import pytest
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from nav_learning.environment import NavigationEnv
from nav_learning.sac_math import Settings
from nav_learning.io_utils import load_config, restore_rng, atomic_json
from nav_learning.checkpoints import save_checkpoint, resolve_checkpoint
from nav_learning.train_sac import create_model, run_training, apply_validation
from nav_learning.evaluation import evaluate_cases
from fake_backend import FakeBackend

torch.set_num_threads(1)


def configuration():
    c = load_config(Path(__file__).parents[1] / "config/sac_training.yaml")
    c["sac"].update(buffer_size=128, learning_starts=4, batch_size=8,
                     policy_kwargs={"net_arch": [16, 16]})
    c["evaluation"].update(every_steps=16, episodes=2)
    c["checkpoint_every_steps"] = 8
    c["curriculum"]["minimum_steps"] = [0]*5
    return c


def trainer_state():
    return {"stage_started": 0, "last_validation": 0, "validation_rounds": [0]*5, "best_scores": {}}


def test_gymnasium_contract_and_observation_ownership():
    backend = FakeBackend()
    env = NavigationEnv(backend=backend)
    check_env(env, warn=True)
    first, _ = env.reset(seed=123)
    saved = first.copy()
    env.step([.2])
    np.testing.assert_array_equal(first, saved)
    assert not np.shares_memory(first, backend.buffer)
    env.close()
    assert backend.closed


def test_timeout_bootstrap_flags_and_terminal_priority():
    backend = FakeBackend()
    env = NavigationEnv(backend=backend)
    env.reset(seed=123)
    env.task = replace(env.task, max_steps=1)
    _, _, terminal, truncated, info = env.step([0])
    assert not terminal and truncated and info["outcome"] == "timeout"
    env.reset(seed=123)
    env.task = replace(env.task, goal=(backend.x, backend.y), max_steps=1)
    backend.forced_nearest = .21
    _, _, terminal, truncated, info = env.step([0])
    assert terminal and not truncated and info["outcome"] == "collision"


def test_real_sac_actor_critics_entropy_and_replay_update():
    env = NavigationEnv(backend=FakeBackend())
    model = create_model(env, configuration(), "cpu")
    actor = [p.detach().clone() for p in model.actor.parameters()]
    critic = [p.detach().clone() for p in model.critic.parameters()]
    entropy = model.log_ent_coef.detach().clone()
    model.learn(32)
    assert model._n_updates > 0 and model.replay_buffer.size() == 32
    assert any(not torch.equal(a,b) for a,b in zip(actor, model.actor.parameters()))
    assert any(not torch.equal(a,b) for a,b in zip(critic, model.critic.parameters()))
    assert not torch.equal(entropy, model.log_ent_coef)
    assert all(torch.isfinite(p).all() for p in model.policy.parameters())
    prior = model.replay_buffer.observations[:32].copy()
    env.backend.buffer[:] = -1
    np.testing.assert_array_equal(prior, model.replay_buffer.observations[:32])
    env.close()


def test_atomic_checkpoint_restores_model_replay_optimizers_and_rng(tmp_path):
    config = configuration()
    env = NavigationEnv(backend=FakeBackend())
    model = create_model(env, config, "cpu")
    model.learn(24)
    save_checkpoint(model, env, tmp_path, config, trainer_state())
    root, state = resolve_checkpoint(tmp_path)
    restore_rng(state["rng"])
    expected = (random.random(), np.random.rand(), float(torch.rand(())))
    other_env = NavigationEnv(backend=FakeBackend())
    restored = SAC.load(root / "model.zip", env=other_env, device="cpu")
    restored.load_replay_buffer(root / "replay.pkl")
    other_env.load_state_dict(state["environment"])
    assert restored.num_timesteps == 24 and restored._n_updates == model._n_updates
    assert restored.replay_buffer.pos == model.replay_buffer.pos
    np.testing.assert_array_equal(restored.replay_buffer.observations[:24], model.replay_buffer.observations[:24])
    for a,b in zip(restored.actor.parameters(), model.actor.parameters()):
        assert torch.equal(a,b)
    assert restored.actor.optimizer.state_dict()["state"].keys() == model.actor.optimizer.state_dict()["state"].keys()
    restore_rng(state["rng"])
    actual = (random.random(), np.random.rand(), float(torch.rand(())))
    assert actual == expected
    restored.learn(8, reset_num_timesteps=False)
    assert restored.num_timesteps == 32 and restored.replay_buffer.size() == 32
    env.close()
    other_env.close()


def test_validation_driver_results_restore_training_state_and_rng(tmp_path):
    config = configuration()
    env = NavigationEnv(backend=FakeBackend())
    model = create_model(env, config, "cpu")
    model.learn(8)
    before = env.state_dict()
    rng = torch.get_rng_state().clone()
    summary = evaluate_cases(model, env, [520000,520001], 0, tmp_path / "evaluation")
    assert summary["complete"] and summary["completed_episodes"] == 2
    assert sum(summary[k] for k in ("success","collision","timeout")) == 2
    assert env.state_dict() == before and torch.equal(rng, torch.get_rng_state())
    assert model._last_obs is None
    assert (tmp_path/"evaluation/evaluation_episodes.csv").exists()
    model.learn(4, reset_num_timesteps=False)
    assert model.num_timesteps == 12
    env.close()


def test_training_driver_checkpoint_and_absolute_budget(tmp_path):
    config = configuration()
    env = NavigationEnv(backend=FakeBackend())
    model = create_model(env, config, "cpu")
    trainer = run_training(env, model, config, tmp_path, 18, trainer_state(), lambda: False)
    root, state = resolve_checkpoint(tmp_path)
    assert state["steps"] == 18 and model.num_timesteps == 18
    assert trainer["validation_rounds"][0] == 1
    assert (tmp_path/"validation_history.csv").exists()
    with pytest.raises(ValueError):
        run_training(env, model, config, tmp_path, 18, trainer, lambda: False)
    env.close()


def test_promotion_uses_complete_validation_and_minimum_steps():
    config = configuration()
    config["curriculum"]["minimum_steps"][0] = 100
    state = trainer_state()
    summary = {"complete": True, "success_rate": 10/12}
    assert not apply_validation(state, 0, 99, summary, config)
    assert apply_validation(state, 0, 100, summary, config)
    summary["complete"] = False
    assert not apply_validation(state, 0, 200, summary, config)


def test_config_rejects_discount_mismatch(tmp_path):
    import yaml
    config = configuration()
    config["sac"]["gamma"] = .99
    path = tmp_path/"bad.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="discounts must match"):
        load_config(path)


def test_main_fresh_and_resume_use_new_model_and_absolute_target(tmp_path, monkeypatch):
    from contextlib import nullcontext
    import yaml
    import nav_learning.train_sac as trainer
    config = configuration()
    config_path = tmp_path/"config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    project = tmp_path/"project"
    project.mkdir()
    run = project/"runs/sac/test_run"
    monkeypatch.setattr(trainer, "simulator_session", lambda _: nullcontext())
    monkeypatch.setattr(trainer, "NavigationEnv", lambda settings, seed_base: NavigationEnv(settings, FakeBackend(), seed_base))
    trainer.main(["--project-dir",str(project),"--config",str(config_path),"--run-dir",str(run),
                  "--target-steps","18","--device","cpu"])
    _, first = resolve_checkpoint(run)
    assert first["steps"] == 18
    assert (project/"runs/sac/latest_run_path.txt").read_text().strip() == str(run)
    trainer.main(["--project-dir",str(project),"--resume",str(run),"--target-steps","26","--device","cpu"])
    _, second = resolve_checkpoint(run)
    assert second["steps"] == 26
    assert second["environment"]["episode_index"] > first["environment"]["episode_index"]
    assert len(list((run/"checkpoints").glob("step_*"))) == 2


def test_ctrl_c_defers_shutdown_saves_then_resumes(tmp_path, monkeypatch):
    from contextlib import nullcontext
    import os
    import signal
    import yaml
    import nav_learning.train_sac as trainer
    config = configuration()
    path = tmp_path/"config.yaml"
    path.write_text(yaml.safe_dump(config))
    project = tmp_path/"project"
    project.mkdir()
    backends = []
    class InterruptingBackend(FakeBackend):
        def transition(self, v, w):
            result = super().transition(v,w)
            os.kill(os.getpid(), signal.SIGINT)
            return result
    def create_env(settings, seed_base):
        backend = InterruptingBackend() if not backends else FakeBackend()
        backends.append(backend)
        return NavigationEnv(settings, backend, seed_base)
    monkeypatch.setattr(trainer,"simulator_session",lambda _:nullcontext())
    monkeypatch.setattr(trainer,"NavigationEnv",create_env)
    run = project/"runs/sac/interrupt_test"
    trainer.main(["--project-dir",str(project),"--config",str(path),"--run-dir",str(run),
                  "--target-steps","18","--device","cpu"])
    _, state = resolve_checkpoint(run)
    assert state["steps"] == 1 and backends[0].closed
    trainer.main(["--project-dir",str(project),"--resume",str(run),"--target-steps","10","--device","cpu"])
    _, state = resolve_checkpoint(run)
    assert state["steps"] == 10 and backends[-1].closed


def test_replay_keeps_timeouts_bootstrappable(tmp_path):
    class OneStepEnv(NavigationEnv):
        def reset(self, **kwargs):
            obs, info = super().reset(**kwargs)
            self.task = replace(self.task, max_steps=1)
            return obs, info
    env = OneStepEnv(backend=FakeBackend())
    model = create_model(env, configuration(), "cpu")
    model.learn(2)
    assert model.replay_buffer.dones[:2].all()
    assert model.replay_buffer.timeouts[:2].all()
    batch = model.replay_buffer.sample(2)
    assert not batch.dones.any()
    env.close()
