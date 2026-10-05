# Priority Inheritance for Reducing Priority Inversion in Real-Time Systems

## Project structure

- `app.py` — Streamlit dashboard
- `scheduler_engine.py` — single source of truth for both scheduling policies
- `simulator.py` — baseline wrapper
- `priority_inheritance.py` — PI wrapper
- `metrics.py` — evaluation metrics
- `risk_model.py` — lightweight Decision Tree risk classifier
- `comparison.py` — CLI comparison report
- `scenarios.py` — fixed experimental scenario
- `models/priority_inversion_risk_model.pkl` — generated locally by `risk_model.py`

## Clean setup

Activate the existing virtual environment, then:

```powershell
pip install -r requirements.txt
```

Train/rebuild the AI model:

```powershell
python -c "from risk_model import train_model; print(train_model())"
```

Check all Python files for syntax errors:

```powershell
python -m py_compile scenarios.py scheduler_engine.py simulator.py priority_inheritance.py metrics.py risk_model.py comparison.py app.py
```

Run the dashboard:

```powershell
streamlit run app.py
```

Open:

```text
http://localhost:8501
```

Then click:

**Run Complete Experiment**

## Expected key results

For the fixed scenario:

- H blocking time: `9 -> 5`
- H response time: `11 -> 7`
- H completion time: `12 -> 8`
- H deadline: `MISSED -> MET`
- CPU utilization: `100% -> 100%`
- Jain fairness: approximately `0.857 -> 0.857`

The CPU utilization and fairness values remain unchanged because the experiment uses a fixed workload and PI changes the execution order rather than the total CPU service required by each task.

## AI note

The risk classifier is intentionally lightweight and educational. Its training labels are generated from domain-informed synthetic scheduling conditions. The model demonstrates how an AI-based early-warning layer can identify high-risk conditions before the scheduler applies Priority Inheritance.
