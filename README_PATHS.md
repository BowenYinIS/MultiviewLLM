# Path Configuration Guide

This project uses environment variables to manage machine-specific data paths, making it portable across different machines.

## Setup

### 1. Configure Your Paths

The `.env` file has been created with default paths. If you need to change them:

1. Open `.env` in the project root (`v2/.env`)
2. Update the paths to match your machine:

```bash
# Example:
RAW_DATA_DIR=/home/yuzhu/chaoyang/projects/Delinquency/delinquency/github/v2/data/raw_data
PROCESSED_DATA_DIR=/home/yuzhu/chaoyang/projects/Delinquency/delinquency/github/v2/data/processed_data
CHECKPOINT_DIR=/home/yuzhu/chaoyang/projects/Delinquency/delinquency/github/v2/checkpoint
```

### 2. Using Paths in Code

Import the `paths` object from `src.config.paths`:

```python
from src.config.paths import paths

# Access directories
print(paths.raw_data_dir)          # Path to raw data directory
print(paths.processed_data_dir)    # Path to processed data directory
print(paths.checkpoint_dir)        # Path to checkpoint directory

# Access specific files
print(paths.act_info)              # Path to act_info.feather
print(paths.sample_transaction)    # Path to sample_transaction.feather

# Use in your code
import polars as pl
df = pl.read_ipc(paths.act_info, memory_map=False)
```

### 3. Adding New Paths

To add new paths to the configuration:

1. Add the environment variable to `.env`:
   ```bash
   NEW_DATA_DIR=/path/to/new/data
   ```

2. Add it to `.env.example` with a placeholder:
   ```bash
   NEW_DATA_DIR=/path/to/your/new/data
   ```

3. Add a property in `src/config/paths.py`:
   ```python
   class DataPaths:
       def __init__(self):
           # ... existing code ...
           self.new_data_dir = Path(os.getenv("NEW_DATA_DIR", "data/new_data"))
   ```

## File Structure

```
v2/
├── .env                    # Machine-specific paths (gitignored)
├── .env.example           # Template for documentation (committed to git)
└── src/config/
    ├── __init__.py
    └── paths.py           # Python module to load and provide paths
```

## Notes

- `.env` is gitignored to prevent committing machine-specific paths
- `.env.example` serves as documentation and should be committed
- All paths are converted to `pathlib.Path` objects for cross-platform compatibility
- Relative paths in `.env` are resolved relative to the project root (`v2/`)
- Absolute paths in `.env` are used as-is

