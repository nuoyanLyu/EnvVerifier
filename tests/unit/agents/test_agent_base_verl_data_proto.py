import logging

import torch

from agentfly.agents.agent_base import BaseAgent


class _FakeDataProto:
    @classmethod
    def from_single_dict(cls, inputs, meta_info=None):
        return {"inputs": inputs, "meta_info": meta_info}


class _TestAgent(BaseAgent):
    @property
    def rewards(self):
        return [0.5, 0.25], {"classification": ["complete", "extra", "unused"]}


def test_get_verl_data_proto_uses_agent_logger_for_mismatched_reward_metadata(monkeypatch):
    agent = _TestAgent.__new__(_TestAgent)
    agent.logger = logging.getLogger("test-agent")

    monkeypatch.setattr(
        "agentfly.agents.agent_base.DataProto",
        _FakeDataProto,
    )

    def tokenize_trajectories(*, return_reward_mask, concatenate_mm_inputs):
        assert return_reward_mask is True
        assert concatenate_mm_inputs is False
        return (
            {"reward_mask": torch.ones(2, 3)},
            [{"group_id": "task-0"}, {"group_id": "task-1"}],
        )

    agent.tokenize_trajectories = tokenize_trajectories

    batch = agent.get_verl_data_proto()

    assert batch["meta_info"] == {"use_agent": True}
    assert batch["inputs"]["uid"].tolist() == ["task-0", "task-1"]
    assert batch["inputs"]["rm_classification"].tolist() == ["complete", "extra"]
