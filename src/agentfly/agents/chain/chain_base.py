import asyncio
import inspect
import json
import logging
import sys
import time
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from termcolor import colored
from tqdm.asyncio import tqdm_asyncio

from ...tools.tool_base import submit_tool_call
from ...utils.monitor import MetricEvent, Monitor, emit, serialize_for_json
from ...utils.timing import Timer
from ...utils.vision import image_to_data_uri
from ..utils.messages import Messages, MessagesList
from .streaming_observer import ConsoleStreamObserver, StreamEvent, StreamEventType

logger = logging.getLogger(__name__)


@dataclass
class Node:
    messages: Messages
    is_terminal: bool = False
    is_pruned: bool = False
    type: Optional[str] = None
    description: str = ""
    observation: str = ""
    observation_code: Optional[str] = None
    parent: Optional["Node"] = None
    children: List["Node"] = field(default_factory=list)

    @property
    def depth(self) -> int:
        return 0 if self.parent is None else self.parent.depth + 1

    def print_node(self, process_id: int = 0) -> None:
        if process_id != 0:
            return
        color_converter = {
            "Thought": "red",
            "Action": "blue",
            "Action Input": "cyan",
            "Final Answer": "green",
            "Reflection": "blue",
        }
        color = color_converter.get(self.type, "white")
        print(colored(f"{self.type}: {self.description}", color=color))
        if self.observation:
            obs = (
                self.observation
                if len(self.observation) < 1536
                else f"{self.observation[:1536]}...(len={len(self.observation)})"
            )
            print(colored(f"Observation: {obs}", color="yellow"))

    def to_json(self, use_messages: bool = False) -> dict:
        json_obj = {
            "is_terminal": self.is_terminal,
            "is_pruned": self.is_pruned,
            "depth": self.depth,
            "type": self.type,
            "description": self.description,
            "messages": self.messages if use_messages else [],
        }
        if self.observation:
            json_obj["observation"] = self.observation
        if self.observation_code is not None:
            json_obj["observation_code"] = self.observation_code
        return json_obj

    def to_json_recursive(self, use_messages: bool = False) -> dict:
        data = self.to_json(use_messages=use_messages)
        data["children"] = [
            child.to_json_recursive(use_messages=use_messages)
            for child in self.children
        ]
        return data


class Chain:
    """
    Manages a sequential chain of nodes (chain-of-thought).
    Each node can have at most one child.
    """

    def __init__(self, info):
        self.root: Optional[Node] = None
        self.info: Dict[str, Any] = info

    def add_node(
        self,
        is_terminal: bool = False,
        is_pruned: bool = False,
        type: Optional[str] = None,
        description: str = "",
        observation: str = "",
        observation_code: Optional[str] = None,
        messages: Optional[List[Any]] = None,
    ) -> Node:
        messages = Messages.from_turns(messages)
        new_node = Node(
            is_terminal=is_terminal,
            is_pruned=is_pruned,
            type=type,
            description=description,
            observation=observation,
            observation_code=observation_code,
            messages=messages,
        )
        if self.root is None:
            self.root = new_node
        else:
            current = self.root
            while len(current.children) > 0:
                current = current.children[0]
            current.children = [new_node]
            new_node.parent = current
        return new_node

    def to_json(self) -> List[dict]:
        chain_json = []
        node = self.root
        while node:
            chain_json.append(node.to_json())
            if node.children:
                node = node.children[0]
            else:
                break
        return chain_json


class ChainRollout:
    """
    Basic class for chain-based rollout. It starts multiple chains and runs them asynchronously.
    """

    def __init__(self):
        self.reset()
        self.chains: Dict[str, Chain] = {}
        self.current_nodes: Dict[str, Node] = {}
        self.timer = Timer()
        self.terminal_status = ["terminal", "finish"]
        self.global_step = 0
        self.finished_chains_count = 0
        self.monitor_info = defaultdict(list)

    def reset(self) -> None:
        self.status_code: str = "continue"
        self.query_count: int = 0  # Number of interactions
        self.total_tokens: int = 0
        self.success_count: int = 0
        self.chains = []
        self.current_nodes = {}

    def _diag_mode(self) -> str:
        return getattr(self, "chain_diagnostics_mode", "key_stages")

    def _diag_enabled_for(self, event_kind: str) -> bool:
        if not getattr(self, "chain_diagnostics_enabled", False):
            return False
        mode = self._diag_mode()
        if mode == "verbose":
            return True
        if mode == "key_stages":
            return True
        if mode == "errors_only":
            return event_kind in {"timeout", "retry", "failure"}
        return False

    def _truncate_for_log(self, value: Any) -> str:
        payload_chars = getattr(self, "chain_diagnostics_payload_chars", 256)
        text = serialize_for_json(value)
        if len(text) <= payload_chars:
            return text
        return f"{text[:payload_chars]}...(len={len(text)})"

    def _message_summary_for_log(self, messages: List[Dict[str, Any]]) -> str:
        if not messages:
            return "[]"
        last_message = messages[-1]
        return self._truncate_for_log(
            {
                "role": last_message.get("role"),
                "content": last_message.get("content"),
                "tool_calls": last_message.get("tool_calls"),
                "tool_name": last_message.get("tool_name"),
            }
        )

    def _diag_log(
        self,
        event: str,
        *,
        event_kind: str = "stage",
        chain_id: Optional[str] = None,
        group_id: Optional[str] = None,
        depth: Optional[int] = None,
        tool_name: Optional[str] = None,
        attempt: Optional[int] = None,
        elapsed_s: Optional[float] = None,
        payload: Any = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self._diag_enabled_for(event_kind):
            return

        fields = [f"event={event}"]
        if chain_id is not None:
            fields.append(f"chain_id={chain_id}")
        if group_id is not None:
            fields.append(f"group_id={group_id}")
        if depth is not None:
            fields.append(f"depth={depth}")
        if tool_name is not None:
            fields.append(f"tool={tool_name}")
        if attempt is not None:
            fields.append(f"attempt={attempt}")
        if elapsed_s is not None:
            fields.append(f"elapsed_s={elapsed_s:.2f}")
        if extra:
            for key, value in extra.items():
                fields.append(f"{key}={serialize_for_json(value)}")
        if payload is not None and self._diag_mode() in {"key_stages", "verbose"}:
            fields.append(f"payload={self._truncate_for_log(payload)}")

        logger.info("[ChainDiagnostics] %s", " ".join(fields))

    async def _run_with_timeout_retries(
        self,
        operation_name: str,
        chain_id: str,
        group_id: Optional[str],
        depth: int,
        func,
        *,
        timeout_s: Optional[float],
        max_retries: int,
        payload: Any = None,
        tool_name: Optional[str] = None,
    ):
        attempts = max_retries + 1
        last_error = None
        for attempt in range(1, attempts + 1):
            started_at = time.time()
            self._diag_log(
                f"{operation_name}_start",
                chain_id=chain_id,
                group_id=group_id,
                depth=depth,
                tool_name=tool_name,
                attempt=attempt,
                payload=payload,
            )
            try:
                if timeout_s is not None and timeout_s > 0:
                    result = await asyncio.wait_for(func(), timeout=timeout_s)
                else:
                    result = await func()
                self._diag_log(
                    f"{operation_name}_success",
                    chain_id=chain_id,
                    group_id=group_id,
                    depth=depth,
                    tool_name=tool_name,
                    attempt=attempt,
                    elapsed_s=time.time() - started_at,
                    payload=result if self._diag_mode() == "verbose" else None,
                )
                return result
            except asyncio.TimeoutError as exc:
                last_error = exc
                self._diag_log(
                    f"{operation_name}_timeout",
                    event_kind="timeout",
                    chain_id=chain_id,
                    group_id=group_id,
                    depth=depth,
                    tool_name=tool_name,
                    attempt=attempt,
                    elapsed_s=time.time() - started_at,
                    payload=payload,
                )
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                self._diag_log(
                    f"{operation_name}_failure",
                    event_kind="failure",
                    chain_id=chain_id,
                    group_id=group_id,
                    depth=depth,
                    tool_name=tool_name,
                    attempt=attempt,
                    elapsed_s=time.time() - started_at,
                    extra={"error": repr(exc)},
                )
                raise

            if attempt < attempts:
                self._diag_log(
                    f"{operation_name}_retry",
                    event_kind="retry",
                    chain_id=chain_id,
                    group_id=group_id,
                    depth=depth,
                    tool_name=tool_name,
                    attempt=attempt + 1,
                    extra={"backoff_s": getattr(self, "chain_retry_backoff_s", 1.0)},
                )
                backoff_s = getattr(self, "chain_retry_backoff_s", 1.0)
                if backoff_s > 0:
                    await asyncio.sleep(backoff_s)

        raise last_error


    @property
    def timing_data(self):
        return self.timer.timing_data

    def to_json(self) -> dict:
        return {
            "finish": [chain.status_code == "success" for chain in self.chains],
            "chains": [chain.to_json() for chain in self.chains],
        }

    def initialize_chains(
        self, messages_list: MessagesList, num_chains: int
    ) -> Tuple[Dict[str, Chain], Dict[str, Node]]:
        chains = {}
        start_nodes = {}
        group_ids = [str(uuid.uuid4()) for _ in range(len(messages_list))]

        for group_idx, messages in enumerate(messages_list):
            group_id = group_ids[group_idx]
            for j in range(num_chains):
                ch = Chain(messages.meta | {"group_id": group_id})
                root = ch.add_node(
                    type="Action Input", messages=deepcopy(messages.messages)
                )

                cid = str(uuid.uuid4())
                chains[cid] = ch
                start_nodes[cid] = root

        return chains, start_nodes

    def get_messages(self) -> List[Any]:
        messages = []
        for id, node in self.current_nodes.items():
            info = self.chains[id].info
            message_item = {}
            message_item["messages"] = node.messages.messages
            message_item.update(info)
            messages.append(message_item)
        return messages

    def validate_run_args(
        self, max_turns: int, num_chains: int, enable_streaming: bool
    ):
        assert max_turns >= 1, "max_turns must be at least 1."
        assert num_chains >= 1, "num_chains must be at least 1."
        for observer in self.streaming_manager.observers:
            if isinstance(observer, ConsoleStreamObserver) and enable_streaming:
                assert num_chains == 1, (
                    "num_chains must be 1 when ConsoleStreamObserver is used."
                )

    async def run_async(
        self,
        messages: List[Dict],
        max_turns: int,
        num_chains: int,
        generation_config: Optional[Dict[str, Any]] = None,
        enable_streaming: bool = False,
    ):
        """
        Run the chain-based rollout with optional streaming support.

        Args:
            max_steps: Maximum number of steps for each chain.
            start_messages: List of messages to start the chains.
            num_chains: Number of chains to run for each message.
            generation_config: Generation configuration dictionary.
            enable_streaming: Whether to enable streaming mode.
            streaming_callback: Optional callback for streaming events.
        """
        self.validate_run_args(max_turns, num_chains, enable_streaming)
        Monitor.ensure_started()
        self.reset()

        # 抽取消息
        messages_list = MessagesList.from_data(messages)
        chains, first_nodes = self.initialize_chains(messages_list, num_chains)
        # 从工具中解析schema，得到工具列表的描述信息
        tool_schemas = [tool.schema for tool in self.tools]

        done_q = asyncio.Queue()
        tasks = [
            asyncio.create_task(
                self._run_single_chain(
                    cid,
                    node,
                    chains[cid],
                    tool_schemas,
                    max_turns=max_turns,
                    generation_config=generation_config,
                    done_queue=done_q,
                    enable_streaming=enable_streaming,
                )
            )
            for cid, node in first_nodes.items()
        ]

        await tqdm_asyncio.gather(*tasks, file=sys.stdout)

        self.chains = {}
        while not done_q.empty():
            cid, chain, node = done_q.get_nowait()
            self.chains[cid] = chain
            self.current_nodes[cid] = node

        self.global_step += 1
        self.monitor_step()

    async def _run_single_chain(
        self,
        chain_id: str,
        first_node: Node,
        chain: Chain,
        tools: List[Dict],
        max_turns: int,
        generation_config: Dict[str, Any],
        done_queue: asyncio.Queue,
        enable_streaming: bool = False,
    ):
        """
        Run a single chain with optional streaming support.
        """
        current_node = first_node
        depth = 0
        have_set_tools = False
        group_id = chain.info.get("group_id")

        self._diag_log(
            "chain_start",
            chain_id=chain_id,
            group_id=group_id,
            depth=depth,
            payload=self._message_summary_for_log(first_node.messages.messages),
        )

        while not current_node.is_terminal and depth < max_turns:
            newest_messages = current_node.messages.copy()

            if not current_node.is_terminal:
                # Generate response
                # new_msg = await self._generate_response(
                #     current_node=current_node,
                #     tools=tools,
                #     depth=depth,
                #     chain_id=chain_id,
                #     generation_config=generation_config,
                #     enable_streaming=enable_streaming,
                # )
                try:
                    new_msg = await self._generate_response(
                        current_node=current_node,
                        tools=tools,
                        depth=depth,
                        chain_id=chain_id,
                        group_id=group_id,
                        generation_config=generation_config,
                        enable_streaming=enable_streaming,
                    )
                except Exception as exc:  # noqa: BLE001
                    timeout_s = getattr(self, "chain_generation_timeout_s", None)
                    if isinstance(exc, asyncio.TimeoutError):
                        failure_text = (
                            f"generation timed out after {timeout_s:.2f}s"
                            if timeout_s is not None
                            else "generation timed out"
                        )
                    else:
                        failure_text = f"generation failed: {type(exc).__name__}: {exc}"
                    new_msg = {
                        "role": "assistant",
                        "content": [{"type": "text", "text": failure_text}],
                        "tool_calls": [],
                        "status": "terminal",
                    }
                    self._diag_log(
                        "generation_final_failure",
                        event_kind="failure",
                        chain_id=chain_id,
                        group_id=group_id,
                        depth=depth,
                        extra={"error": failure_text},
                    )

                newest_messages.append(new_msg)
                thought_node = chain.add_node(
                    type="Thought",
                    messages=newest_messages.copy(),
                    description=new_msg.get("content", ""),
                )
                thought_node.is_terminal = (
                    new_msg.get("status", "continue") in self.terminal_status
                )
                current_node = thought_node

                # Check if the thought node is terminal - if so, break the loop
                if current_node.is_terminal:
                    break

            # Handle tool calls
            if current_node.messages[-1].get("tool_calls"):
                for tool_call in current_node.messages[-1]["tool_calls"]:
                    # result = await self._execute_tool_call(
                    #     tool_call,
                    #     newest_messages,
                    #     chain,
                    #     chain_id,
                    #     depth,
                    #     have_set_tools,
                    #     enable_streaming,
                    # )
                    try:
                        result = await self._execute_tool_call(
                            tool_call,
                            newest_messages,
                            chain,
                            chain_id,
                            group_id,
                            depth,
                            have_set_tools,
                            enable_streaming,
                        )
                    except Exception as exc:  # noqa: BLE001
                        timeout_s = getattr(self, "chain_tool_timeout_s", None)
                        tool_name = tool_call["function"]["name"]
                        tool_input = tool_call["function"]["arguments"]
                        if isinstance(exc, asyncio.TimeoutError):
                            failure_text = (
                                f"tool '{tool_name}' timed out after {timeout_s:.2f}s"
                                if timeout_s is not None
                                else f"tool '{tool_name}' timed out"
                            )
                        else:
                            failure_text = (
                                f"tool '{tool_name}' failed: {type(exc).__name__}: {exc}"
                            )
                        result = {
                            "name": tool_name,
                            "arguments": tool_input,
                            "observation": failure_text,
                            "status": "terminal",
                        }
                        self._diag_log(
                            "tool_final_failure",
                            event_kind="failure",
                            chain_id=chain_id,
                            group_id=group_id,
                            depth=depth,
                            tool_name=tool_name,
                            extra={"error": failure_text},
                        )

                    have_set_tools = True

                    # Create action input node
                    action_input_node = chain.add_node(
                        type="Action Input",
                        messages=newest_messages.copy(),
                        description=result.get("arguments", ""),
                    )

                    # Process observation
                    observation = result["observation"]

                    action_input_node.observation = observation
                    action_input_node.observation_code = result["status"]

                    new_content = [{"type": "text", "text": observation}]
                    # Handle multi-modal outputs
                    if "image" in result:
                        image = result["image"]
                        image_base64 = image_to_data_uri(image)
                        new_content.append({"type": "image", "image": image_base64})

                    newest_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call["id"],
                            "tool_name": result["name"],
                            "content": new_content,
                        }
                    )
                    action_input_node.messages = newest_messages.copy()
                    action_input_node.is_terminal = (
                        result["status"] in self.terminal_status
                    )
            else:
                # No tool calls, chain is finished
                break

            current_node = action_input_node

            depth += 1

        # Finalize chain
        await self._finalize_chain(chain_id, chain, current_node, depth)
        await done_queue.put((chain_id, chain, current_node))

        self.finished_chains_count += 1
        message_info = chain.info
        self._diag_log(
            "chain_end",
            chain_id=chain_id,
            group_id=group_id,
            depth=depth,
            payload=self._message_summary_for_log(current_node.messages.messages),
            extra={"reward": chain.info.get("reward")},
        )
        self.monitor_chain(trajectory=current_node.messages.messages, info=message_info)

    async def _generate_response(
        self,
        current_node,
        tools,
        depth,
        chain_id,
        group_id,
        generation_config,
        enable_streaming,
    ):
        """Generate response with optional streaming support."""
        payload = self._message_summary_for_log(current_node.messages.messages)

        async def run_once():
            return await self._generate_response_once(
                current_node=current_node,
                tools=tools,
                depth=depth,
                chain_id=chain_id,
                generation_config=generation_config,
                enable_streaming=enable_streaming,
            )

        return await self._run_with_timeout_retries(
            "generation",
            chain_id,
            group_id,
            depth,
            run_once,
            timeout_s=getattr(self, "chain_generation_timeout_s", None),
            max_retries=getattr(self, "chain_generation_max_retries", 0),
            payload=payload,
        )
        # if enable_streaming:
        #     # Emit generation start event
        #     await self.streaming_manager.emit_event(
        #         StreamEvent(
        #             event_type=StreamEventType.LLM_GENERATION_START,
        #             chain_id=chain_id,
        #             timestamp=time.time(),
        #             data={"depth": depth},
        #             step=depth,
        #             depth=depth,
        #         )
        #     )

        #     # Check if we have streaming capabilities
        #     has_streaming = False
        #     if hasattr(self, "generate_streaming"):
        #         has_streaming = True
        #     elif hasattr(self, "llm_engine") and hasattr(
        #         self.llm_engine, "generate_streaming"
        #     ):
        #         has_streaming = True

        #         # Create a wrapper to use the LLM engine's streaming
        #         async def generate_streaming_wrapper(messages_list, **kwargs):
        #             async for chunk in self.llm_engine.generate_streaming(
        #                 messages_list, **kwargs
        #             ):
        #                 yield chunk

        #         self.generate_streaming = generate_streaming_wrapper

        #     if has_streaming:
        #         # Collect full response from streaming
        #         full_response = ""
        #         async for chunk in self.generate_streaming(
        #             [current_node.messages.messages], tools=tools, **generation_config
        #         ):
        #             await self.streaming_manager.emit_event(
        #                 StreamEvent(
        #                     event_type=StreamEventType.LLM_GENERATION_CHUNK,
        #                     chain_id=chain_id,
        #                     timestamp=time.time(),
        #                     data={"content": chunk},
        #                     step=depth,
        #                     depth=depth,
        #                 )
        #             )
        #             # chunk is the whole generated text
        #             full_response = chunk

        #         logger.debug(
        #             f"[ChainRollout._generate_response] full_response: {full_response}"
        #         )

        #         # Emit generation end event
        #         await self.streaming_manager.emit_event(
        #             StreamEvent(
        #                 event_type=StreamEventType.LLM_GENERATION_END,
        #                 chain_id=chain_id,
        #                 timestamp=time.time(),
        #                 data={"full_response": full_response},
        #                 step=depth,
        #                 depth=depth,
        #             )
        #         )

        #         # Parse response
        #         new_msg = self.parse([full_response], tools=tools)
        #         # new_msg = self.parse([full_response], )
        #         return new_msg[0]
        #     else:
        #         # Fallback to non-streaming generation
        #         responses = await self.generate_async(
        #             [current_node.messages.messages], tools=tools, **generation_config
        #         )
        #         # new_msg = self.parse(responses)
        #         new_msg = self.parse(responses, tools=tools)


        #         # Emit a single chunk event for the full response
        #         full_response = new_msg[0].get("content", "")
        #         if isinstance(full_response, list) and len(full_response) > 0:
        #             # Handle case where content is a list of content blocks
        #             if (
        #                 isinstance(full_response[0], dict)
        #                 and "text" in full_response[0]
        #             ):
        #                 full_response = full_response[0]["text"]
        #             else:
        #                 full_response = str(full_response)
        #         elif not isinstance(full_response, str):
        #             full_response = str(full_response)

        #         await self.streaming_manager.emit_event(
        #             StreamEvent(
        #                 event_type=StreamEventType.LLM_GENERATION_CHUNK,
        #                 chain_id=chain_id,
        #                 timestamp=time.time(),
        #                 data={"content": full_response},
        #                 step=depth,
        #                 depth=depth,
        #             )
        #         )

        #         # Emit generation end event
        #         await self.streaming_manager.emit_event(
        #             StreamEvent(
        #                 event_type=StreamEventType.LLM_GENERATION_END,
        #                 chain_id=chain_id,
        #                 timestamp=time.time(),
        #                 data={"full_response": full_response},
        #                 step=depth,
        #                 depth=depth,
        #             )
        #         )

        #         return new_msg[0]
        # else:
        #     # Non-streaming generation
        #     responses = await self.generate_async(
        #         [current_node.messages.messages], tools=tools, **generation_config
        #     )
        #     # new_msg = self.parse(responses)
        #     new_msg = self.parse(responses, tools=tools)
        #     return new_msg[0]

    async def _generate_response_once(
        self, current_node, tools, depth, chain_id, generation_config, enable_streaming
    ):
        """Generate a single response attempt with optional streaming support."""
        if enable_streaming:
            # Emit generation start event
            await self.streaming_manager.emit_event(
                StreamEvent(
                    event_type=StreamEventType.LLM_GENERATION_START,
                    chain_id=chain_id,
                    timestamp=time.time(),
                    data={"depth": depth},
                    step=depth,
                    depth=depth,
                )
            )

            # Check if we have streaming capabilities
            has_streaming = False
            if hasattr(self, "generate_streaming"):
                has_streaming = True
            elif hasattr(self, "llm_engine") and hasattr(
                self.llm_engine, "generate_streaming"
            ):
                has_streaming = True

                # Create a wrapper to use the LLM engine's streaming
                async def generate_streaming_wrapper(messages_list, **kwargs):
                    async for chunk in self.llm_engine.generate_streaming(
                        messages_list, **kwargs
                    ):
                        yield chunk

                self.generate_streaming = generate_streaming_wrapper

            if has_streaming:
                # Collect full response from streaming
                full_response = ""
                async for chunk in self.generate_streaming(
                    [current_node.messages.messages], tools=tools, **generation_config
                ):
                    await self.streaming_manager.emit_event(
                        StreamEvent(
                            event_type=StreamEventType.LLM_GENERATION_CHUNK,
                            chain_id=chain_id,
                            timestamp=time.time(),
                            data={"content": chunk},
                            step=depth,
                            depth=depth,
                        )
                    )
                    # chunk is the whole generated text
                    full_response = chunk

                logger.debug(
                    f"[ChainRollout._generate_response] full_response: {full_response}"
                )

                # Emit generation end event
                await self.streaming_manager.emit_event(
                    StreamEvent(
                        event_type=StreamEventType.LLM_GENERATION_END,
                        chain_id=chain_id,
                        timestamp=time.time(),
                        data={"full_response": full_response},
                        step=depth,
                        depth=depth,
                    )
                )

                # Parse response
                new_msg = self.parse([full_response], tools=tools)
                return new_msg[0]
            else:
                # Fallback to non-streaming generation
                responses = await self.generate_async(
                    [current_node.messages.messages], tools=tools, **generation_config
                )
                new_msg = self.parse(responses, tools=tools)

                # Emit a single chunk event for the full response
                full_response = new_msg[0].get("content", "")
                if isinstance(full_response, list) and len(full_response) > 0:
                    # Handle case where content is a list of content blocks
                    if (
                        isinstance(full_response[0], dict)
                        and "text" in full_response[0]
                    ):
                        full_response = full_response[0]["text"]
                    else:
                        full_response = str(full_response)
                elif not isinstance(full_response, str):
                    full_response = str(full_response)

                await self.streaming_manager.emit_event(
                    StreamEvent(
                        event_type=StreamEventType.LLM_GENERATION_CHUNK,
                        chain_id=chain_id,
                        timestamp=time.time(),
                        data={"content": full_response},
                        step=depth,
                        depth=depth,
                    )
                )

                # Emit generation end event
                await self.streaming_manager.emit_event(
                    StreamEvent(
                        event_type=StreamEventType.LLM_GENERATION_END,
                        chain_id=chain_id,
                        timestamp=time.time(),
                        data={"full_response": full_response},
                        step=depth,
                        depth=depth,
                    )
                )

                return new_msg[0]
        else:
            # Non-streaming generation
            responses = await self.generate_async(
                [current_node.messages.messages], tools=tools, **generation_config
            )
            new_msg = self.parse(responses, tools=tools)
            return new_msg[0]


    async def _execute_tool_call(
        self,
        tool_call,
        newest_messages,
        chain,
        chain_id,
        group_id,
        depth,
        have_set_tools,
        enable_streaming,
    ):
        """Execute a tool call with optional streaming support."""
        # breakpoint()
        tool_name = tool_call["function"]["name"]
        tool_input = tool_call["function"]["arguments"]

        # Set up tools if needed
        if not have_set_tools:
            await self.set_tools(chain_id, chain.info)
            have_set_tools = True

        # Execute tool call
        # result = await submit_tool_call(
        #     tool_name, tool_input, id=chain_id, allowed_tool_names=self.tool_names
        # )
        async def run_once():
            return await submit_tool_call(
                tool_name, tool_input, id=chain_id, allowed_tool_names=self.tool_names
            )

        result = await self._run_with_timeout_retries(
            "tool",
            chain_id,
            group_id,
            depth,
            run_once,
            timeout_s=getattr(self, "chain_tool_timeout_s", None),
            max_retries=getattr(self, "chain_tool_max_retries", 0),
            payload={"arguments": tool_input},
            tool_name=tool_name,
        )

        if enable_streaming:
            # Emit tool observation event
            tool_data = {
                "tool_name": tool_name,
                "observation": result["observation"],
                "status": result["status"],
            }
            if "image" in result:
                tool_data["image"] = result["image"]
            await self.streaming_manager.emit_event(
                StreamEvent(
                    event_type=StreamEventType.TOOL_OBSERVATION,
                    chain_id=chain_id,
                    timestamp=time.time(),
                    data=tool_data,
                    step=depth,
                    depth=depth,
                )
            )

        return result

    async def _finalize_chain(self, chain_id, chain, current_node, depth):
        """Finalize the chain with reward calculation and cleanup."""
        if self._reward_fn is not None:
            trajectory = current_node.messages.messages
            final_response = self.extract_final_response(trajectory)
            other_args = {
                k: v
                for k, v in chain.info.items()
                if k not in ["final_response", "trajectory", "id"]
            }

            # TODO: move the reward calculation to reward module
            reward = self._reward_fn(
                final_response=final_response,
                **other_args,
                trajectory=trajectory,
                id=chain_id,
            )
            if inspect.iscoroutine(reward):
                reward = await reward

            chain.info["reward"] = reward
        else:
            chain.info["reward"] = None

        await self.release_resources(chain_id)

    async def release_resources(self, id: str) -> None:
        for tool in self.tools:
            await tool.release(id=id)
        if self._reward_fn is not None:
            await self._reward_fn.release(id=id)

    async def set_tools(self, id: str, env_args: Dict[str, Any]) -> None:
        for tool in self.tools:
            await tool.set_env(id, env_args)

    def monitor_step(self) -> None:
        messages = self.get_messages()
        avg_turns = 0
        avg_tool_calls = 0
        # avg_response_length = 0
        tool_calls_by_name = defaultdict(int)

        for message in messages:
            for msg in message["messages"]:
                if msg["role"] == "assistant":
                    avg_turns += 1
                if msg["role"] == "tool":
                    avg_tool_calls += 1
                    tool_call_name = msg["tool_name"]
                    tool_calls_by_name[tool_call_name] += 1

        avg_turns /= len(messages)
        avg_tool_calls /= len(messages)

        ent = MetricEvent(
            kind="scalar",
            name="Agent/rollout/step",
            value=self.global_step,
            x=self.global_step,
            x_name="Agent/rollout/step",
        )
        emit(ent)

        evt = MetricEvent(
            kind="scalar",
            name="Agent/rollout/avg_turns",
            value=avg_turns,
            x=self.global_step,
            x_name="Agent/rollout/step",
        )
        emit(evt)

        evt = MetricEvent(
            kind="scalar",
            name="Agent/rollout/avg_tool_calls",
            value=avg_tool_calls,
            x=self.global_step,
            x_name="Agent/rollout/step",
        )
        emit(evt)

        for tool_name, tool_call_count in tool_calls_by_name.items():
            evt = MetricEvent(
                kind="scalar",
                name=f"Agent/rollout/tool_calls/{tool_name}",
                value=tool_call_count,
                x=self.global_step,
                x_name="Agent/rollout/step",
            )
            emit(evt)

        evt = MetricEvent(
            kind="scalar",
            name="Agent/rollout/step",
            value=self.global_step,
            x=self.global_step,
            x_name="Agent/rollout/step",
        )
        emit(evt)

        sample_message_json = json.dumps(serialize_for_json(messages[0]), indent=2)
        evt = MetricEvent(
            kind="text",
            name="Agent/rollout/sample_message",
            value=sample_message_json,
            x=self.global_step,
            x_name="Agent/rollout/step",
        )
        emit(evt)

        for k, v in self.monitor_info.items():
            if k != "Agent/chains":  # We don't log number of chains
                evt = MetricEvent(
                    kind="list",
                    name=k,
                    value=v,
                    x=self.monitor_info["Agent/chains"],
                )
                emit(evt)

    def monitor_chain(self, trajectory, info) -> None:
        self.monitor_info["Agent/chains"].append(self.finished_chains_count)
        for tool in self.tools:
            if tool.is_stateful and tool.pool_size > 0:
                self.monitor_info[f"Agent/Tool/{tool.name}/used_env_size"].append(
                    tool.used_env_size
                )

        # We only log the trajectory to local jsonl file, for wandb much bandwidth is needed
        evt = MetricEvent(
            sinks=["jsonl"],
            kind="text",
            name="Agent/rollout/trajectory",
            value=json.dumps(serialize_for_json(trajectory), indent=2),
            x=self.global_step,
            x_name="Agent/rollout/step",
        )
        emit(evt)

        evt = MetricEvent(
            sinks=["jsonl"],
            kind="text",
            name="Agent/rollout/info",
            value=json.dumps(serialize_for_json(info), indent=2),
            x=self.global_step,
            x_name="Agent/rollout/step",
        )
        emit(evt)
