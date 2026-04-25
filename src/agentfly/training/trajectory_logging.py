import dataclasses
import json
import os
from pathlib import Path


@dataclasses.dataclass
class ValidationTrajectoryLogger:
    project_name: str | None = None
    experiment_name: str | None = None

    def log(self, loggers, records, step, key="val/trajectories"):
        if not records:
            return

        if "wandb" in loggers:
            self.log_records_to_wandb(records, step, key)
        if "swanlab" in loggers:
            self.log_records_to_swanlab(records, step, key)
        if "mlflow" in loggers:
            self.log_records_to_mlflow(records, step, key)
        if "clearml" in loggers:
            self.log_records_to_clearml(records, step, key)
        if "tensorboard" in loggers:
            self.log_records_to_tensorboard(records, step, key)
        if "vemlp_wandb" in loggers:
            self.log_records_to_vemlp_wandb(records, step, key)

    @staticmethod
    def _stringify_value(value):
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        try:
            return json.dumps(value, ensure_ascii=False, indent=2)
        except TypeError:
            return str(value)

    def _normalize_records(self, records):
        normalized = []
        for record in records:
            normalized.append(
                {key: self._stringify_value(value) for key, value in record.items()}
            )
        return normalized

    @staticmethod
    def _get_columns(records):
        columns = []
        for record in records:
            for key in record:
                if key not in columns:
                    columns.append(key)
        return columns

    def log_records_to_vemlp_wandb(self, records, step, key):
        from volcengine_ml_platform import wandb as vemlp_wandb

        self._log_records_to_wandb(records, step, key, vemlp_wandb)

    def log_records_to_wandb(self, records, step, key):
        import wandb

        self._log_records_to_wandb(records, step, key, wandb)

    def _log_records_to_wandb(self, records, step, key, wandb):
        records = self._normalize_records(records)
        columns = ["step", *self._get_columns(records)]

        if not hasattr(self, "record_tables"):
            self.record_tables = {}

        table_key = key.replace("/", "__")
        if table_key not in self.record_tables:
            self.record_tables[table_key] = wandb.Table(columns=columns)

        existing_table = self.record_tables[table_key]
        new_table = wandb.Table(columns=columns, data=existing_table.data)

        for record in records:
            row = [step, *[record.get(column, "") for column in columns[1:]]]
            new_table.add_data(*row)

        wandb.log({key: new_table}, step=step)
        self.record_tables[table_key] = new_table

    def log_records_to_swanlab(self, records, step, key):
        import swanlab

        records = self._normalize_records(records)
        headers = ["step", *self._get_columns(records)]
        rows = [[step, *[record.get(column, "") for column in headers[1:]]] for record in records]

        table = swanlab.echarts.Table()
        table.add(headers=headers, rows=rows)
        swanlab.log({key: table}, step=step)

    def log_records_to_mlflow(self, records, step, key):
        import tempfile

        import mlflow

        records = self._normalize_records(records)

        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                artifact_name = key.replace("/", "_")
                output_path = Path(tmp_dir, f"{artifact_name}_step{step}.json")
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(records, f, ensure_ascii=False, indent=2)
                mlflow.log_artifact(str(output_path))
        except Exception as e:
            print(f"WARNING: save {key} to mlflow failed with error {e}")

    def log_records_to_clearml(self, records, step, key):
        import clearml
        import pandas as pd

        task: clearml.Task | None = clearml.Task.current_task()
        if task is None:
            return

        records = self._normalize_records(records)
        task.get_logger().report_table(
            series=key,
            title="Validation",
            table_plot=pd.DataFrame.from_records(records),
            iteration=step,
        )

    def log_records_to_tensorboard(self, records, step, key):
        if not hasattr(self, "writer"):
            from torch.utils.tensorboard import SummaryWriter

            if self.project_name and self.experiment_name:
                default_dir = os.path.join("tensorboard_log", self.project_name, self.experiment_name)
            else:
                default_dir = "tensorboard_log"
            tensorboard_dir = os.environ.get("TENSORBOARD_DIR", default_dir)
            os.makedirs(tensorboard_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=tensorboard_dir)

        records = self._normalize_records(records)
        text_content = f"**{key} - Step {step}**\n\n"
        for idx, record in enumerate(records):
            text_content += f"### Record {idx + 1}\n"
            for record_key, record_value in record.items():
                text_content += f"**{record_key}:** {record_value}\n\n"
            text_content += "---\n\n"

        self.writer.add_text(key, text_content, step)
        self.writer.flush()
