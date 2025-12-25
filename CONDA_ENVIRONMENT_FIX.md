# AutoTrain Conda Environment Fix

## Problem
When running autotrain in a specific conda environment, the application was trying to find the `autotrain` module using the system Python interpreter (`/usr/bin/python3`) instead of the conda environment's Python interpreter. This caused the following error:

```
/usr/bin/python3: Error while finding module specification for 'autotrain'
ModuleNotFoundError: No module named 'autotrain'
```

## Root Cause
When using `accelerate launch` for distributed training, the accelerate launcher spawns child processes that may not inherit the conda environment properly. The code was relying on the system's default Python interpreter instead of explicitly specifying the Python executable from the conda environment.

## Solution Implemented

### 1. **src/autotrain/commands.py**

#### Changes Made:
- **Added `import sys`** to access the current Python executable
- **Added helper function `_get_python_executable()`** - Returns `sys.executable` to get the current environment's Python path
- **Added helper function `_prepend_python_to_command(cmd)`** - Converts accelerate commands from:
  ```python
  ["accelerate", "launch", ...]
  ```
  to:
  ```python
  ["/path/to/conda/python", "-m", "accelerate.commands.launch", ...]
  ```
- **Updated `launch_command()` function signature** - Added optional parameter `use_python_executable=True` to allow disabling this feature if needed
- **Added Python executable prepending** at the end of `launch_command()` to ensure all `accelerate` commands use the conda environment's Python

#### Key Code Addition:
```python
def _get_python_executable():
    """Get the Python executable path from the current environment."""
    return sys.executable

def _prepend_python_to_command(cmd):
    """Prepend the Python executable to accelerate commands."""
    if cmd and cmd[0] == "accelerate":
        return [_get_python_executable(), "-m", "accelerate.commands.launch"] + cmd[2:]
    return cmd
```

### 2. **src/autotrain/utils.py**

#### Changes Made:
- **Updated `run_training()` function signature** to handle both legacy and new interfaces:
  - Legacy: `run_training(command)` - accepts a command string or list
  - New: `run_training(params=..., task_id=..., wait=...)` - accepts training parameters and task ID
- **Added automatic command generation** from params using `launch_command()` when params and task_id are provided
- **Added PYTHONPATH setup** in the environment variables to ensure Python can find all modules in the conda environment
- **Changed subprocess call to use list format** instead of shell=True for accelerate commands, which is more reliable
- **Added PYTHONUNBUFFERED flag** to ensure real-time log output

#### Key Code Addition:
```python
# Create environment with proper Python path
env = os.environ.copy()
env["PYTHONPATH"] = os.pathsep.join(sys.path)
env["PYTHONUNBUFFERED"] = "1"

# Execute command using list format (not shell)
if isinstance(cmd, list):
    process = subprocess.Popen(
        cmd,
        shell=False,  # Important: Use list format
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        env=env,
    )
```

### 3. **src/autotrain/backends/local.py**

#### Changes Made:
- **Added PYTHONPATH to environment variables** when running Automatic Speech Recognition (ASR) training:
  ```python
  env["PYTHONPATH"] = os.pathsep.join(sys.path)
  ```

## How It Works

### Before (Broken):
1. User activates conda environment: `conda activate myenv`
2. autotrain calls `accelerate launch -m autotrain.trainers...`
3. accelerate spawns subprocess without inheriting conda environment
4. Subprocess uses `/usr/bin/python3` (system Python) to find modules
5. **Error**: `autotrain` module not found (not installed in system Python)

### After (Fixed):
1. User activates conda environment: `conda activate myenv`
2. autotrain calls `python -m accelerate.commands.launch -m autotrain.trainers...`
3. The Python path is explicitly set to the conda environment's Python
4. Subprocess inherits PYTHONPATH and other environment variables
5. **Success**: Modules found in conda environment's site-packages

## Configuration Values That Were Updated

| File | Setting | Old Value | New Value |
|------|---------|-----------|-----------|
| commands.py | Python executable | Implicit (uses PATH) | Explicit via `sys.executable` |
| commands.py | Command format | `["accelerate", "launch", ...]` | `[python_path, "-m", "accelerate.commands.launch", ...]` |
| utils.py | Function signature | `run_training(command)` | `run_training(command=None, params=None, task_id=None, wait=False)` |
| utils.py | PYTHONPATH | Not set | Explicitly set from `sys.path` |
| utils.py | subprocess shell | `shell=True` | `shell=False` for list commands |
| local.py | Environment setup | Only PYTHONUNBUFFERED | PYTHONUNBUFFERED + PYTHONPATH |

## Testing the Fix

To verify the fix is working:

1. **Activate your conda environment:**
   ```bash
   conda activate your_env
   ```

2. **Check Python executable:**
   ```bash
   python -c "import sys; print(sys.executable)"
   ```
   Should output: `/path/to/your/env/bin/python` (not `/usr/bin/python3`)

3. **Verify autotrain is installed:**
   ```bash
   python -c "import autotrain; print(autotrain.__file__)"
   ```
   Should output a path inside your conda environment

4. **Run autotrain training:**
   - Check the logs for: `Python executable: /path/to/conda/env/bin/python`
   - This confirms the correct Python executable is being used

## Backward Compatibility

These changes are **fully backward compatible**:
- The `run_training()` function accepts the old `command` parameter
- Commands can still be strings or lists
- Existing code using `run_training(command)` will continue to work
- The new interface (`run_training(params=..., task_id=...)`) is additional functionality

## Environment Variables Set by autotrain

When autotrain runs training, it now sets:

| Variable | Value | Purpose |
|----------|-------|---------|
| PYTHONPATH | All paths from `sys.path` in current environment | Ensures Python finds all modules in conda environment |
| PYTHONUNBUFFERED | 1 | Disables output buffering for real-time log streaming |
| All existing vars | Inherited from parent | Maintains parent environment configuration |

## Potential Issues and Solutions

### Issue: Still getting "module not found" error
**Solution**: 
- Verify conda environment is activated: `conda activate your_env`
- Check Python path: `python -c "import sys; print(sys.path)"`
- Reinstall autotrain: `pip install -e .` (in development mode)

### Issue: subprocess.Popen errors with list commands
**Solution**:
- Ensure all command elements are strings
- Check for None or invalid values in command list
- Fallback to shell=True if needed (less secure but more compatible)

### Issue: PYTHONPATH conflicts
**Solution**:
- Clear conda environment and rebuild
- Remove any conflicting system Python packages
- Use conda to manage all Python packages instead of system pip

## Files Modified

1. `/home/ritesh/autotrain-advanced/src/autotrain/commands.py` - Command generation with Python executable support
2. `/home/ritesh/autotrain-advanced/src/autotrain/utils.py` - Enhanced run_training function
3. `/home/ritesh/autotrain-advanced/src/autotrain/backends/local.py` - Added PYTHONPATH to ASR training

## References

- [Python sys.executable documentation](https://docs.python.org/3/library/sys.html#sys.executable)
- [subprocess.Popen documentation](https://docs.python.org/3/library/subprocess.html#subprocess.Popen)
- [PYTHONPATH documentation](https://docs.python.org/3/using/cmdline.html#envvar-PYTHONPATH)
- [Conda environment documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/concepts/environments.html)
- [Accelerate documentation](https://huggingface.co/docs/accelerate/)
