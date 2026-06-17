import asyncio

from agentfly.agents.chain.chain_base import Chain, Node
from agentfly.agents.utils.messages import Messages


def test_node_creation():
    node = Node(
        is_terminal=False,
        type="Thought",
        description="This is a test thought",
        observation="Test observation",
        messages=Messages.from_turns([{"role": "user", "content": "test"}]),
    )

    assert node.is_terminal == False
    assert node.type == "Thought"
    assert node.description == "This is a test thought"
    assert node.observation == "Test observation"
    assert node.depth == 0
    assert len(node.children) == 0


def test_node_to_json():
    node = Node(
        is_terminal=False,
        type="Action",
        description="google_search",
        observation="Test result",
        messages=[{"role": "user", "content": "test"}],
    )

    json_data = node.to_json(use_messages=True)

    assert json_data["is_terminal"] == False
    assert json_data["type"] == "Action"
    assert json_data["description"] == "google_search"
    assert json_data["observation"] == "Test result"
    assert len(json_data["messages"]) == 1
    assert json_data["messages"][0]["role"] == "user"


def test_chain_creation():
    chain = Chain(info={"question": "test question"})

    assert chain.info["question"] == "test question"
    assert chain.root is None


def test_chain_add_node():
    chain = Chain(info={"question": "test question"})

    # Add root node
    root = chain.add_node(
        type="Thought",
        description="Initial thought",
        messages=Messages.from_turns([{"role": "user", "content": "test"}]),
    )

    assert chain.root == root
    assert root.type == "Thought"
    assert root.description == "Initial thought"

    # Add child node
    child = chain.add_node(
        type="Action",
        description="google_search",
        messages=Messages.from_turns([{"role": "user", "content": "test"}]),
    )

    assert len(root.children) == 1
    assert root.children[0] == child
    assert child.parent == root
    assert child.depth == 1


def test_chain_to_json():
    chain = Chain(info={"question": "test question"})
    chain.add_node(
        type="Thought",
        description="Initial thought",
        messages=Messages.from_turns([{"role": "user", "content": "test"}]),
    )
    chain.add_node(
        type="Action",
        description="google_search",
        messages=Messages.from_turns([{"role": "user", "content": "test"}]),
    )

    json_data = chain.to_json()

    assert len(json_data) == 2
    assert json_data[0]["type"] == "Thought"
    assert json_data[1]["type"] == "Action"


def test_rollout_infra_failure_is_resampled_without_reward(monkeypatch):
    from agentfly.agents.chain import chain_base

    class _StreamingManager:
        observers = []

    class _Tool:
        name = "fake_tool"
        schema = {"type": "function", "function": {"name": "fake_tool"}}
        is_stateful = False
        pool_size = 0

    class _Rollout(chain_base.ChainRollout):
        def __init__(self):
            super().__init__()
            self.tools = [_Tool()]
            self.tool_names = ["fake_tool"]
            self.streaming_manager = _StreamingManager()
            self.chain_rollout_max_resample_attempts = 1
            self.set_tools_calls = 0
            self.reward_calls = 0

            async def reward_fn(**kwargs):
                self.reward_calls += 1
                return {"reward": 1.0}

            self._reward_fn = reward_fn
            self._reward_fn.release = self._release_reward

        async def _release_reward(self, id, success=True):
            return None

        async def generate_async(self, messages_list_or_inputs, tools=None, **kwargs):
            return ["call fake tool"]

        def parse(self, responses, tools=None):
            return [
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": response}],
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "fake_tool", "arguments": "{}"},
                        }
                    ],
                    "status": "continue",
                }
                for response in responses
            ]

        async def set_tools(self, id, env_args):
            self.set_tools_calls += 1
            if self.set_tools_calls == 1:
                raise chain_base.RolloutInfraError("environment setup failed")

        async def release_resources(self, id, success=True):
            return None

        def extract_final_response(self, messages):
            return messages[-1]["content"][0]["text"]

    async def fake_submit_tool_call(*args, **kwargs):
        return {
            "name": "fake_tool",
            "arguments": {},
            "observation": "ok",
            "status": "terminal",
        }

    monkeypatch.setattr(chain_base, "submit_tool_call", fake_submit_tool_call)

    rollout = _Rollout()
    asyncio.run(
        rollout.run_async(
            [{"role": "user", "content": "hello"}],
            max_turns=1,
            num_chains=1,
            generation_config={},
        )
    )

    trajectories = rollout.get_messages()
    assert len(trajectories) == 1
    assert rollout.aborted_chains_count == 1
    assert rollout.reward_calls == 1
    assert "rollout_aborted" not in trajectories[0]
    assert [message["role"] for message in trajectories[0]["messages"]] == [
        "user",
        "assistant",
        "tool",
    ]
    assert trajectories[0]["messages"][-1]["content"][0]["text"] == "ok"
