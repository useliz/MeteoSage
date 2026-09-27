# MeteoSage
This is the official implementation of MeteoSage, an agentic framework that connects observation, analysis, forecasting, and weather services through a stateful, evidence-grounded workflow. 

We also introduced WxFlowBench[https://huggingface.co/datasets/when665/WxFlowBench], a benchmark grounded in multi-source observations, operational products, and real-world meteorological reports. 

## Code organization

| Module | Entry points | Responsibility |
| --- | --- | --- |
| Startup and composition | [Startup](src/weather_agent/startup.py), [scientific assets](src/weather_agent/installed_science.py), [public composition](src/weather_agent/public_composition.py) | Construct task context and assemble scientific capabilities. |
| Agent execution | [Episode runner](src/weather_agent/episode_runner.py), [model decision adapter](src/weather_agent/hosted_agent_policy.py) | Build model requests, process decisions, and execute operations. |
| State and evidence | [Run state](src/weather_agent/run_state.py), [working state](src/weather_agent/working_state.py), [evidence](src/weather_agent/evidence.py) | Maintain plans, observations, contextual state, and evidence references. |
| Scientific operations | [Code execution](src/weather_agent/coderun_operation.py), [forecast operations](src/weather_agent/forecast_operations.py), [knowledge operations](src/weather_agent/knowledge_operations.py) | Bind public actions to their inputs, execution contracts, and results. |
| Report delivery | [Report schema](src/weather_agent/t3_t4/report_schema.py), [report submission](src/weather_agent/t3_t4/report_final.py) | Define report arguments and connect submissions with evidence and validation. |
| Trajectory memory | [Memory records](src/weather_agent/trajectory_memory.py), [retrieval](src/weather_agent/trajectory_memory_retrieval.py), [selection](src/weather_agent/trajectory_memory_selection.py), [generation](src/weather_agent/trajectory_memory_generation.py) | Retrieve, select, deliver, and generate task-relevant advice. |

The [code map](docs/code-map.md) describes the module interfaces and reading order. The [file manifest](docs/source-manifest.json) lists paths, module descriptions, and content hashes.

## Licence
MIT
