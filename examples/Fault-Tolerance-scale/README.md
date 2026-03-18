# Instructions for using the vllm scale_down script
## Start the vllm service
| Parameter Variable  | Default Value                      | Description                               |
|---------------------|------------------------------------|-------------------------------------------|
| HOST                | 0.0.0.0                            | Set host address                          |
| PORT                | 8006                               | vLLM API service port                     |
| DATA_PARALLEL_SIZE  | 4                                  | Data Parallel (DP) size                   |
| REDUNDANT_EXPERTS   | 0                                  | Number of redundant experts               |
| FAULT_PORT          | 22867                              | The port to use for external fault notify |
| LOCAL_MODEL_PATH    | /models/Qwen3-30B-A3B-W8A8         | Local model file path                     |
| MODEL_NAME          | /qwen-ai/Qwen3-30B-A3B-W8A8        | Model name                                |
| GLOO_TIMEOUT        | 30                                 | Gloo communication group timeout          |
To launch the vLLM service, run the startup script with the following command:
```bash
bash serve_qwen.sh --dp 8 --re 48
```

## Send scale command
| Parameter Variable         | Default Value | Description                                      |
|----------------------------|---------------|--------------------------------------------------|
| host                       | 0.0.0.0       | Vllm serveAPI server host                        |
| port                       | 8006          | VLLM API service port                            |
| timeout                    | 30            | Fault recovery timeout                           |
| exclude-dp-ranks           | 0             | The dp_ranks that will be excluded (scaled down) |
| external-fault-notify-port | 22867         | The port to use for external fault notify        |

For service scale-down, use the same script with identical parameters:
```python
python scale_down.py --host 192.1.1.1 --port 8006 --exclude-dp-ranks 3
```
If you want to trigger a pause by killing the process
```python
python scale_down.py --host 192.1.1.1 --port 8006 --exclude-dp-ranks 3 --kill-process
```

You can use dcmi.py to detect machine fault codes and trigger scale-down when a machine malfunctions.

```python
python dcmi.py --port 8006 
```