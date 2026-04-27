import json
from collections import defaultdict

import numpy as np

from ..verl.protocol import DataProto, pad_dataproto_to_divisor, unpad_dataproto
from ..verl.trainer.ppo.metric_utils import process_validation_metrics
from ..verl.trainer.ppo.ray_trainer import RayPPOTrainer
from .trajectory_logging import ValidationTrajectoryLogger


class TrajectoryLoggingRayPPOTrainer(RayPPOTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._install_swanlab_log_filter()
        self.validation_trajectory_logger = ValidationTrajectoryLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

    def _install_swanlab_log_filter(self) -> None:
        trainer_loggers = self.config.trainer.logger
        if "swanlab" not in trainer_loggers:
            return

        try:
            import swanlab
        except ImportError:
            return

        if getattr(swanlab, "_agentfly_log_filter_installed", False):
            return

        original_log = swanlab.log
        filtered_metric_keys = {
            "training/global_step",
            "training/epoch",
        }

        def filtered_log(data=None, *args, **kwargs):
            if isinstance(data, dict):
                data = {
                    key: value for key, value in data.items() if key not in filtered_metric_keys
                }
                if not data:
                    return None
            return original_log(data, *args, **kwargs)

        swanlab.log = filtered_log
        swanlab._agentfly_log_filter_installed = True

    @staticmethod
    def _extract_message_text(message: dict) -> str:
        content = message.get("content", "")
        if isinstance(content, list):
            text_parts = []
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "text":
                        text_parts.append(part.get("text") or "")
                    else:
                        text_parts.append(json.dumps(part, ensure_ascii=False))
                else:
                    text_parts.append(str(part))
            return "\n".join(part for part in text_parts if part).strip()
        if content is None:
            return ""
        return str(content)

    @staticmethod
    def _stringify_for_log(value):
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        try:
            return json.dumps(value, ensure_ascii=False, indent=2)
        except TypeError:
            return str(value)

    def _build_validation_trajectory_record(
        self,
        trajectory: dict,
        score: float,
        reward_extra: dict,
        ground_truth=None,
        data_source: str | None = None,
    ) -> dict:
        messages = trajectory.get("messages", [])

        model_input_messages = []
        assistant_outputs = []
        tool_outputs = []
        seen_assistant = False

        for message in messages:
            role = message.get("role")
            if role == "assistant":
                seen_assistant = True
                assistant_outputs.append(self._extract_message_text(message))
            elif role == "tool":
                tool_outputs.append(
                    {
                        "tool_name": message.get("tool_name"),
                        "tool_call_id": message.get("tool_call_id"),
                        "content": self._extract_message_text(message),
                    }
                )

            if not seen_assistant:
                model_input_messages.append(message)

        first_model_output = assistant_outputs[0] if assistant_outputs else ""
        final_output = assistant_outputs[-1] if assistant_outputs else ""

        return {
            "uid": trajectory.get("group_id", ""),
            "data_source": data_source or "",
            "ground_truth": self._stringify_for_log(ground_truth),
            "model_input": self._stringify_for_log(model_input_messages),
            "first_model_output": first_model_output,
            "model_outputs": self._stringify_for_log(assistant_outputs),
            "tool_outputs": self._stringify_for_log(tool_outputs),
            "final_output": final_output,
            "reward": score,
            "reward_extra": self._stringify_for_log(reward_extra),
            "trajectory": self._stringify_for_log(messages),
        }

    def _maybe_log_val_trajectories(self, trajectory_records):
        records_to_log = self.config.trainer.get("log_val_trajectories", 4)
        if records_to_log == 0 or not trajectory_records:
            return

        records = list(trajectory_records)
        records.sort(key=lambda record: str(record.get("model_input", "")))
        rng = np.random.RandomState(42)
        rng.shuffle(records)
        records = records[:records_to_log]

        self.validation_trajectory_logger.log(
            self.config.trainer.logger,
            records,
            self.global_steps,
            key="val/trajectories",
        )

    def _validate(self):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []
        sample_uids = []
        sample_trajectory_records = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            input_messages = test_batch.non_tensor_batch["messages"].tolist()
            sample_inputs.extend([self._stringify_for_log(messages) for messages in input_messages])

            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
            ]
            sample_gts.extend(ground_truths)

            test_gen_batch = self._get_gen_batch(test_batch)
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            size_divisor = (
                self.actor_rollout_wg.world_size
                if not self.async_rollout_mode
                else self.config.actor_rollout_ref.rollout.agent.num_workers
            )
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            if not self.async_rollout_mode:
                raise NotImplementedError("Validation trajectory logging currently expects async rollout mode.")
            else:
                self.agent_wrapper.set_llm_engine(self.async_rollout_manager, self.tokenizer, self.processor)
                self.run_on_bg(
                    self.agent_wrapper.run(
                        max_turns=self.config.agent.max_turns,
                        messages=test_gen_batch_padded.non_tensor_batch["messages"],
                        num_chains=1,
                        generation_config=self.config.agent.generation_config,
                    )
                )
                test_output_gen_batch_padded = self.agent_wrapper.get_verl_data_proto()

            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
            batch_trajectories = self.agent_wrapper.trajectories[: len(test_output_gen_batch)]

            sample_uids.extend(test_output_gen_batch.non_tensor_batch["uid"])

            assistant_outputs = []
            for trajectory in batch_trajectories:
                assistant_messages = [
                    self._extract_message_text(message)
                    for message in trajectory.get("messages", [])
                    if message.get("role") == "assistant"
                ]
                assistant_outputs.append(assistant_messages[-1] if assistant_messages else "")
            sample_outputs.extend(assistant_outputs)

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            if self.val_reward_fn is None:
                raise ValueError("val_reward_fn must be provided for validation.")
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            batch_reward_extra_infos = result.get("reward_extra_info", {})
            if "reward_extra_info" in result:
                for key, values in result["reward_extra_info"].items():
                    reward_extra_infos_dict[key].extend(values)

            batch_data_sources = test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0])
            if isinstance(batch_data_sources, np.ndarray):
                batch_data_sources = batch_data_sources.tolist()
            else:
                batch_data_sources = list(batch_data_sources)

            for i, trajectory in enumerate(batch_trajectories):
                reward_extra = {}
                for key, values in batch_reward_extra_infos.items():
                    if i < len(values):
                        reward_extra[key] = values[i]

                sample_trajectory_records.append(
                    self._build_validation_trajectory_record(
                        trajectory=trajectory,
                        score=scores[i],
                        reward_extra=reward_extra,
                        ground_truth=ground_truths[i] if i < len(ground_truths) else None,
                        data_source=batch_data_sources[i] if i < len(batch_data_sources) else "unknown",
                    )
                )

            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(batch_data_sources)

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)
        self._maybe_log_val_trajectories(sample_trajectory_records)

        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                gts=sample_gts,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, values in reward_extra_infos_dict.items():
            assert len(values) == 0 or len(values) == len(sample_scores), (
                f"{key_info}: {len(values)=}, {len(sample_scores)=}"
            )

        data_sources = np.concatenate(data_source_lst, axis=0)
        data_src2var2metric2val = process_validation_metrics(data_sources, sample_uids, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    metric_dict[f"{metric_sec}/{data_source}/{var_name}/{metric_name}"] = metric_val

        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        return metric_dict
