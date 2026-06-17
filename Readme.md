# Learning
Just want to improve my deep learning skills

## uv workspace notes

This repo can hold many small deep learning subprojects. Use a `uv` workspace at the repo root when the subprojects can share one Python version and compatible dependency versions.

Recommended layout:

```text
learn-deep-learning/
├─ pyproject.toml
├─ uv.lock
├─ Readme.md
├─ handwritten-digit-classifier/
│  ├─ pyproject.toml
│  └─ data.py
└─ another-project/
   ├─ pyproject.toml
   └─ main.py
```

### Create the root workspace

From the repo root:

```powershell
uv init
```

Then edit the root `pyproject.toml` so it includes workspace members:

```toml
[tool.uv.workspace]
members = ["*/"]
exclude = [".venv", ".git"]
```

### Create a new subproject

From the repo root:

```powershell
mkdir my-new-project
cd my-new-project
uv init
cd ..
```

When `uv init` is run inside an existing workspace, uv can add the new project as a workspace member.

### Add dependencies to one subproject

Run this from the repo root:

```powershell
uv add --package handwritten-digit-classifier torch torchvision numpy matplotlib
```

Use the package name from that subproject's `pyproject.toml`.

### Sync dependencies

From the repo root:

```powershell
uv sync
```

This creates or updates `.venv` and `uv.lock`.

### Run a subproject script

From the repo root:

```powershell
uv run --package handwritten-digit-classifier python handwritten-digit-classifier/data.py
```

### Manual virtual environment activation on Windows

Most of the time, prefer `uv run`. If you want to activate the environment manually:

```powershell
uv sync
.venv\Scripts\activate
```

Then run Python commands normally:

```powershell
python handwritten-digit-classifier/data.py
```

### When not to use a workspace

If subprojects need different Python versions or conflicting dependency versions, make each folder an independent `uv` project instead:

```powershell
cd handwritten-digit-classifier
uv init
uv add torch torchvision numpy matplotlib
uv run python data.py
```

### Daily working cmd

```bash
uv lock

uv sync --package handwritten-digit-classifier
```