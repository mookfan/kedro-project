from kedro.framework.hooks import hook_impl
from memory_profiler import memory_usage
import logging

from typing import Any, Dict
import mlflow
import mlflow.sklearn
from kedro.pipeline.node import Node

def _normalise_mem_usage(mem_usage):
    # memory_profiler < 0.56.0 returns list instead of float
    return mem_usage[0] if isinstance(mem_usage, (list, tuple)) else mem_usage

class MemoryProfilingHooks:
    def __init__(self):
        self._logger.info("Memory profiling hooks initialised")
        self._mem_usage = {}

    @property
    def _logger(self):
        return logging.getLogger(self.__class__.__name__)

    @hook_impl
    def before_dataset_loaded(self, dataset_name: str) -> None:
        before_mem_usage = memory_usage(
            -1,
            interval=0.1,
            max_usage=True,
            retval=True,
            include_children=True,
        )
        before_mem_usage = _normalise_mem_usage(before_mem_usage)
        self._mem_usage[dataset_name] = before_mem_usage

    @hook_impl
    def after_dataset_loaded(self, dataset_name: str) -> None:
        after_mem_usage = memory_usage(
            -1,
            interval=0.1,
            max_usage=True,
            retval=True,
            include_children=True,
        )
        # memory_profiler < 0.56.0 returns list instead of float
        after_mem_usage = _normalise_mem_usage(after_mem_usage)

        self._logger.info(
            "Loading %s consumed %2.2fMiB memory",
            dataset_name,
            after_mem_usage - self._mem_usage[dataset_name],
        )

class ModelTrackingHooks:
    """Namespace for grouping all model-tracking hooks with MLflow together."""

    @hook_impl
    def before_pipeline_run(self, run_params: Dict[str, Any]) -> None:
        """Hook implementation to start an MLflow run
        with the session_id of the Kedro pipeline run.
        """
        # TODO: Make this configurable
        # mlflow.start_run(run_name=run_params["session_id"])
        mlflow.log_params(run_params)

    @hook_impl
    def after_node_run(
        self, node: Node, outputs: Dict[str, Any], inputs: Dict[str, Any]) -> None:
        """Hook implementation to add model tracking after some node runs.
        In this example, we will:
        * Log the parameters after the data splitting node runs.
        * Log the model after the model training node runs.
        * Log the model's metrics after the model evaluating node runs.
        """
        if node._func_name == "split_data":
            params_name = f"params:{node.namespace}.input_options"
            mlflow.log_params(
                {"split_data_ratio": inputs[params_name]["test_size"]}
            )

        elif node._func_name == "train_model":
            params_name = f"params:{node.namespace}.model_options"
            model = outputs[node._outputs]
            mlflow.sklearn.log_model(model, "model")
            mlflow.log_params(inputs[params_name])

    @hook_impl
    def after_pipeline_run(self) -> None:
        """Hook implementation to end the MLflow run
        after the Kedro pipeline finishes.
        """
        mlflow.end_run()