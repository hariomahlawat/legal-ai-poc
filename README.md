# Legal AI PoC

## Why `apps/ui/main.py` Fails
The Streamlit command in the prompt points at `apps/ui/main.py`, but this repository does not contain that file. The UI entrypoint is `apps/ui/streamlit_app.py`, so Streamlit reports "File does not exist" when given `apps/ui/main.py`.

## How to Run the Streamlit UI
Run Streamlit against the actual entrypoint file and bind it to the desired host/port:

```bash
streamlit run apps/ui/streamlit_app.py --server.port 8501 --server.address 0.0.0.0
```

If you are on Windows PowerShell, use forward slashes for the path as shown above.
