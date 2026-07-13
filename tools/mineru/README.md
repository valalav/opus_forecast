# MinerU ingestion for SIRENA-KBR

This folder keeps MinerU isolated from the main forecasting environment.

## Installed local environments

V100-friendly pipeline environment:

```bash
/home/valalav/_projects/.venvs/mineru-sirena-v100
```

Full environment with Gradio/API/VLM extras:

```bash
/home/valalav/_projects/.venvs/mineru-sirena
```

Both currently use MinerU 3.4.2.

Check versions:

```bash
/home/valalav/_projects/.venvs/mineru-sirena-v100/bin/mineru --version
/home/valalav/_projects/.venvs/mineru-sirena/bin/mineru --version
```

Recreate the V100 pipeline environment if needed:

```bash
uv venv /home/valalav/_projects/.venvs/mineru-sirena-v100 --python 3.12
uv pip install --python /home/valalav/_projects/.venvs/mineru-sirena-v100/bin/python \
  torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
  --index-url https://download.pytorch.org/whl/cu126
uv pip install --python /home/valalav/_projects/.venvs/mineru-sirena-v100/bin/python \
  -r tools/mineru/requirements-v100-pipeline.txt
```

Recreate the full environment if needed:

```bash
uv venv /home/valalav/_projects/.venvs/mineru-sirena --python 3.12
uv pip install --python /home/valalav/_projects/.venvs/mineru-sirena/bin/python -r tools/mineru/requirements-mineru.txt
```

Note: `mineru[all]` currently installs `vllm 0.20.2`, which pins `torch 2.11.0+cu130`.
That PyTorch build does not include `sm70` kernels for Tesla V100, so use the
`mineru-sirena-v100` environment for local GPU pipeline parsing on this host.

## Batch parsing

Russian/Cyrillic PDF batch, conservative pipeline backend:

```bash
MINERU_BIN=/home/valalav/_projects/.venvs/mineru-sirena/bin/mineru \
python3 tools/mineru/batch_mineru.py "ibved/*.pdf" \
  --backend pipeline \
  --method auto \
  --lang cyrillic \
  --device cpu \
  --output-root archive/results/mineru/ibved_cpu
```

V100 GPU pipeline:

```bash
MINERU_BIN=/home/valalav/_projects/.venvs/mineru-sirena-v100/bin/mineru \
python3 tools/mineru/batch_mineru.py "ibved/*.pdf" \
  --backend pipeline \
  --method auto \
  --lang cyrillic \
  --output-root archive/results/mineru/ibved
```

Higher-accuracy hybrid/VLM path for compatible GPUs or remote backends:

```bash
MINERU_BIN=/home/valalav/_projects/.venvs/mineru-sirena/bin/mineru \
python3 tools/mineru/batch_mineru.py "metod/*.pdf" \
  --backend hybrid-engine \
  --effort medium \
  --output-root archive/results/mineru/metod
```

The hybrid/VLM command is for non-V100 GPUs supported by the full `torch 2.11`
stack, or for a remote HTTP backend. It is not the recommended local path on
this V100 host.

Add an Obsidian-ready export folder:

```bash
MINERU_BIN=/home/valalav/_projects/.venvs/mineru-sirena-v100/bin/mineru \
python3 tools/mineru/batch_mineru.py "ibved/*.pdf" \
  --output-root archive/results/mineru/ibved \
  --obsidian-dir /home/valalav/_projects/sirena-kbr/_workspace/obsidian/mineru
```

The wrapper writes:

- `manifest.csv` and `manifest.jsonl`
- one output subfolder per source document
- optional Obsidian folder with a small index note and copied raw MinerU output

## Web UI

```bash
/home/valalav/_projects/.venvs/mineru-sirena/bin/mineru-gradio \
  --server-name 0.0.0.0 \
  --server-port 7860 \
  --api-url http://127.0.0.1:8000
```

Open locally at `http://127.0.0.1:7860` or from LAN at
`http://192.168.11.226:7860`.

For LAN use, keep backend set to `pipeline` on this V100 host and leave the
http-client `server_url` field empty. MinerU blocks `server_url` while the API
is bound to `0.0.0.0` by default, which is the safer behavior.

## API

```bash
/home/valalav/_projects/.venvs/mineru-sirena-v100/bin/mineru-api \
  --host 0.0.0.0 \
  --port 8000
```

Open locally at `http://127.0.0.1:8000/docs` or from LAN at
`http://192.168.11.226:8000/docs`.

Synchronous LAN API test:

```bash
curl -X POST http://192.168.11.226:8000/file_parse \
  -F "files=@ibved/info-stat-11-2025/001-011 титул и содержание.pdf" \
  -F "backend=pipeline" \
  -F "parse_method=auto" \
  -F "lang_list=cyrillic" \
  -F "start_page_id=0" \
  -F "end_page_id=0" \
  -F "return_md=true" \
  -F "return_content_list=true"
```

Asynchronous LAN API flow:

```bash
curl -X POST http://192.168.11.226:8000/tasks \
  -F "files=@ibved/info-stat-11-2025/001-011 титул и содержание.pdf" \
  -F "backend=pipeline" \
  -F "parse_method=auto" \
  -F "lang_list=cyrillic" \
  -F "start_page_id=0" \
  -F "end_page_id=0" \
  -F "return_md=true"

curl http://192.168.11.226:8000/tasks/<task_id>
curl http://192.168.11.226:8000/tasks/<task_id>/result
```
