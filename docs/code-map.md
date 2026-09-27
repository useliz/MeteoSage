# Code map

## Task startup and capability composition

Start with [`startup.py`](../src/weather_agent/startup.py). `Deployment` describes the installed task environment; `build_task` constructs the task and calls `ScientificAssets.assemble_answer` in [`installed_science.py`](../src/weather_agent/installed_science.py).

[`public_composition.py`](../src/weather_agent/public_composition.py) assembles the public operation interface. [`scientific_composition.py`](../src/weather_agent/scientific_composition.py) connects scientific source and tool registrations with the execution resources. [`action_contracts.py`](../src/weather_agent/action_contracts.py) defines operation, preparation, failure, and effect contracts; [`action_operations.py`](../src/weather_agent/action_operations.py) binds registered operation owners to that interface.

## Agent decisions and execution

[`episode_runner.py`](../src/weather_agent/episode_runner.py) defines `Turn`, `Submission`, and `EpisodeRunner`. Follow the runner to see how available operations, decisions, observations, recovery, and final outputs fit together.

[`hosted_agent_policy.py`](../src/weather_agent/hosted_agent_policy.py) implements `HostedDecisionAdapter`, request construction, response interpretation, and provider error handling. [`agent_policy.py`](../src/weather_agent/agent_policy.py) provides `ScenarioDecisionAdapter` for deterministic decisions expressed through the same public operation interface.

## State and evidence

- [`run_state.py`](../src/weather_agent/run_state.py) models plans, evidence requirements, attempts, observations, and execution state.
- [`evidence.py`](../src/weather_agent/evidence.py) defines dataset and variable descriptors, spatial and temporal selections, and evidence records.
- [`working_state.py`](../src/weather_agent/working_state.py) builds the working-state view used by the agent.
- [`working_state_delivery.py`](../src/weather_agent/working_state_delivery.py) packages state and notebook content for the next decision.

## Scientific operation interfaces

- [`coderun_operation.py`](../src/weather_agent/coderun_operation.py) constructs code-execution choices from eligible artifacts, authorized input selections, and execution contracts.
- [`forecast_operations.py`](../src/weather_agent/forecast_operations.py) defines forecast combination descriptors, argument ownership, and execution adapters.
- [`knowledge_operations.py`](../src/weather_agent/knowledge_operations.py) exposes knowledge discovery and reading through the operation interface.

## Report submission

[`t3_t4/report_schema.py`](../src/weather_agent/t3_t4/report_schema.py) builds the report argument schema. [`t3_t4/report_intent.py`](../src/weather_agent/t3_t4/report_intent.py) preserves original report arguments and records submission intent. [`t3_t4/report_final.py`](../src/weather_agent/t3_t4/report_final.py) exposes `final:submit-answer`, connects report sealing with validation results, and completes the task state with the typed output.

## Trajectory memory

| File | Responsibility |
| --- | --- |
| [`trajectory_memory.py`](../src/weather_agent/trajectory_memory.py) | Advisory records, authorized context, recall traces, and memory interfaces. |
| [`trajectory_memory_retrieval.py`](../src/weather_agent/trajectory_memory_retrieval.py) | Semantic and candidate retrieval, reranking, and retrieval RPC clients. |
| [`trajectory_memory_selection.py`](../src/weather_agent/trajectory_memory_selection.py) | Contextual selection, response parsing, advice packing, and selection attempt records. |
| [`trajectory_memory_consumption.py`](../src/weather_agent/trajectory_memory_consumption.py) | Pre-final state, pending-report context, and advice delivery in the next turn. |
| [`trajectory_memory_generation.py`](../src/weather_agent/trajectory_memory_generation.py) | Memory generation configuration, workflow execution, and candidate validation. |
| [`t3_t4/memory_binding.py`](../src/weather_agent/t3_t4/memory_binding.py) | Memory configuration validation and task-node binding. |

Memory prompt templates are organized by their interaction point:

- [Active search](../src/weather_agent/prompts/memory_prefinal/active-search-purpose.txt)
- [Pre-final selection](../src/weather_agent/prompts/memory_prefinal/pre-final-selection.txt)
- [Pre-final advice](../src/weather_agent/prompts/memory_prefinal/pre-final-notice.txt)
- [Report submission](../src/weather_agent/prompts/memory_prefinal/report-final-purpose.txt)
- [Report advice](../src/weather_agent/prompts/memory_prefinal/report-pre-final-notice.txt)
