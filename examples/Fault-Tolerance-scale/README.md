# Instructions for using the vllm scale_down script

This directory contains example scripts for vLLM Ascend Fault Tolerance feature, demonstrating how to start the vLLM service and monitor/handle NPU faults

## File Description

- `serve_qwen.sh` - Launch vLLM service with fault tolerance
- `scale_down.sh` -Monitor NPU status and send scale-down commands

## Prerequisites

1. **Hardware Requirements**
    - huawei Ascend NPU devices
    - DCMI library installed (default path: `/usr/local/dcmi/libdcmi.so`)

2. **Software Requirements**
    - vLLM Ascend installed

## Quick Start

### 1.Start vLLM Service

```bash
# using default configuration(4 DP ranks)
bash serve_qwen.sh

# Custom parameters
bash serve_qwen.sh --dp8 --re 48 --port 8006
```

**Parameters:**

| Parameter Variable | Default Value               | Description                               |
|--------------------|-----------------------------|-------------------------------------------|
| HOST               | 0.0.0.0                     | Set host address                          |
| PORT               | 8006                        | vLLM API service port                     |
| DATA_PARALLEL_SIZE | 4                           | Data Parallel (DP) size                   |
| REDUNDANT_EXPERTS  | 0                           | Number of redundant experts               |
| FAULT_PORT         | 22867                       | The port to use for external fault notify |
| LOCAL_MODEL_PATH   | nytopop/Qwen3-30B-A3B.w8a8  | Local model file path                     |
| MODEL_NAME         | /qwen-ai/Qwen3-30B-A3B-W8A8 | Model name                                |
| GLOO_TIMEOUT       | 30                          | Gloo communication group timeout          |

### 2. Start Scale-Down Monitor
```bash
# Using default configuration
python scale_down.py

# Custom parameters
python scale_down.py --host localhost --port 8006 --npu-ids 0,1,2,3 --interval-time 3
```

**Parameters:**

| Parameter Variable         | Default Value | Description                               |
|----------------------------|---------------|-------------------------------------------|
| host                       | 0.0.0.0       | Vllm serveAPI server host                 |
| port                       | 8006          | VLLM API service port                     |
| recovery_timeout           | 30            | Fault recovery timeout (seconds)          |
| interval_time              | 3             | Interval for polling NPU status (seconds) |
| external-fault-notify-port | 22867         | The port to use for external fault notify |
| npu-ids                    | 0-15          | Comma-separated list of NPU IDS to use    |

## How It Works

### Monitoring Mechanism

`scale_down.py` runs two monitoring threads simultaneously:

1. **vLLM Fault ListenerThread** (`start_monitor_engine_status`)
    - Subscribes to vLLM fault events via ZMQ
    - Sends scale-down command immediately when engine faults are detected

2. **Hardware Fault Detection Thread** (`monitor_machine_fault`)
    - Polls NPU hardware status via DCMI interface
    - Triggers scale-down when fault codes detected (e.g., `0X40f84e00` for card drop)

### Scale-Down Flow

```text
Fault Detected
    ↓
Send pause command (pause request processing for fault DP ranks)
    ↓
Send descale command (notify vLLM to remove fault DP ranks)
    ↓
Update active_npus list (remove fault NPUS from available list)
```


## Complete Example

**Scenario: 8 NPUS deployment, tolerating up to NPU failure**

```bash
# Terminal 1: Start service
bash serve_qwen.sh --dp 8 --re 48 --fault_port 22876

# Terminal 2: Start monitor
python scale_down.py --npu-ids 0,1,2,3,4,5,6,7 --interval-time 3
```

## Important Notes

1. **NPU ID Consistency**
   - The `--dp` parameter in `serve_qwen.sh` specifies the number of NPUs used
   - The `--npu-ids` in `serve_qwen.sh` must match `--dp`, e.g., `--dp 8` corresponds to `--npu-ids 0,1,2,3,4,5,6,7`
   
2. **Port Consistency**
   - The `--fault_port` in `serve_qwen.sh` must match `--external-fault-notify-port` in `scale_down.py` 

   