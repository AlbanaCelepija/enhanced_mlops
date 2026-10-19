## Running the GUI

From the repository root:

1. Install the GUI dependencies:
   ```bash
   pip install -r guided_ui/requirements.txt
   OR
   uv venv
   source .venv/bin/activate 
   uv pip install -r guided_ui/requirements.txt
   ```

2. Install the `temlops` library from the `framework` folder:
   ```bash
   pip install -e framework
   ```

3. Set the `FAIROPS_ONTOLOGY_PATH` environment variable to point to the `fairops.ttl` file of the [fairops ontology](https://github.com/ai-unibo/fairops/tree/main/docs). The GUI reads this variable to populate the interface (Application Domain, AI Task, AI Type of Use, Fairness Concerns, Notions, Metrics and Mitigation Techniques) from the ontology. `indiv.ttl` must live in the same folder as `fairops.ttl`, since it is loaded from there automatically.

   Copy `.env.example` to `.env` and fill in the path:
   ```bash
   cp .env.example .env
   ```
   ```bash
   FAIROPS_ONTOLOGY_PATH=/absolute/path/to/fairops/docs/fairops.ttl
   ```

4. Launch the Streamlit app:
   ```bash
   streamlit run guided_ui/Main.py
   OR
   uv run streamlit run guided_ui/Main.py
   ```
