import shelve
import json
import inspect
import time
import uuid
import functools
from pathlib import Path
import contextlib
from typing import Any, Dict, Optional, Iterator, Callable, TypeVar, Union, cast, List
import traceback
import pandas as pd
import dill

T = TypeVar('T', bound=Callable[..., Any])

shelve.Pickler = dill.Pickler
shelve.Unpickler = dill.Unpickler

def log_parameters(
    func: Optional[Callable] = None, 
    *,
    include: Optional[list[str]] = None,
    exclude: Optional[list[str]] = None,
    **metadata
) -> Union[Callable[[T], T], T]:
    """
    Decorator that logs function parameters and creates a trace ID to link all logs
    within the function call.
    """
    def _decorator(f: T) -> T:
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            try:
                # Get function signature and bind arguments
                sig = inspect.signature(f)
                bound_args = sig.bind(*args, **kwargs)
                bound_args.apply_defaults()
                
                # Get all parameters
                params = dict(bound_args.arguments)
                
                # Extract and validate logger
                logger = params.get('logger')
                if not isinstance(logger, ExperimentLogger):
                    raise ValueError("Missing or invalid logger parameter")
                params.pop('logger')
                
                # Filter parameters based on include/exclude lists
                if include is not None:
                    params = {k: v for k, v in params.items() if k in include}
                if exclude is not None:
                    params = {k: v for k, v in params.items() if k not in exclude}
                
                # Generate trace ID for this function call
                trace_id = str(uuid.uuid4())
                
                # Log parameters with trace ID
                logger.log(
                    params,
                    trace_id=trace_id,
                    event='function_entry',
                    function_name=f.__name__,
                    **metadata
                )
                
            except Exception as e:
                raise RuntimeError(f"Error in parameter logging: {str(e)}") from e

            with logger._trace_context(trace_id):
                try:
                    return f(*args, **kwargs)
                except Exception as function_exception:
                    traceback.print_exc()
                    logger.log(function_exception, function_name=f.__name__)
                    raise
            
        return cast(T, wrapper)
    
    if func is None:
        return _decorator
    return _decorator(func)


def _objects_store_path(log_directory):
    """Resolve purpose-named result stores while retaining legacy log support."""
    directory = Path(log_directory)
    renamed = directory / 'experiment_objects'
    if any((directory / ('experiment_objects' + suffix)).exists()
           for suffix in ('', '.dat', '.dir', '.db')):
        return renamed
    return directory / 'objects'


class ExperimentLogger:
    """A stateful logger that links logs within function calls using trace IDs."""
    
    def __init__(self, 
                 log_directory: Union[str, Path] = 'experiment_logs', 
                 **metadata: Any) -> None:
        """Initialize the logger with a directory and base metadata."""
        self.log_dir = Path(log_directory)
        self.log_dir.mkdir(exist_ok=True, parents=True)
        
        self.objects_path = str(_objects_store_path(self.log_dir))
        renamed_metadata = self.log_dir / 'experiment_events.jsonl'
        self.metadata_path = renamed_metadata if renamed_metadata.exists() else self.log_dir / 'metadata.jsonl'
        self.base_metadata = metadata
        
        # Stack for trace IDs (supports nested function calls)
        self._trace_stack: list[str] = []
        
        if not self.metadata_path.exists():
            self.metadata_path.touch()

    @contextlib.contextmanager
    def _trace_context(self, trace_id: str):
        """Context manager for tracking the current function's trace ID."""
        self._trace_stack.append(trace_id)
        try:
            yield
        finally:
            self._trace_stack.pop()
    
    def _get_caller_info(self) -> Dict[str, Any]:
        """Get information about the actual calling function."""
        frame = inspect.currentframe()
        if frame is None:
            return {}
            
        caller = frame.f_back
        if caller is None:
            return {}
            
        # Skip logger internal frames and decorators
        while caller:
            code = caller.f_code
            if (code.co_filename == __file__ and 
                code.co_name in ['log', '_trace_context', '_decorator', 'wrapper']):
                caller = caller.f_back
            else:
                break
                
        if caller is None:
            return {}
            
        return {
            'function_name': caller.f_code.co_name,
        }

    def _get_variable_name(self, obj: Any, frame_locals: Dict[str, Any]) -> Optional[str]:
        """Get the variable name for an object in the given frame locals."""
        try:
            obj_id = id(obj)
            names = [
                name for name, value in reversed(list(frame_locals.items())) 
                if id(value) == obj_id
            ]
            return names[0] if names else None
        except Exception:
            return None

    def log(self, obj: Any, **metadata: Any) -> None:
        """Log an object with metadata, trace stack, and variable name."""
        try:
            timestamp = time.time()
            obj_id = f"{timestamp:.7f}"
            
            # Get caller information
            caller_info = self._get_caller_info()
            
            # Get variable name
            if frame := inspect.currentframe():
                if caller := frame.f_back:
                    if var_name := self._get_variable_name(obj, caller.f_locals):
                        caller_info['variable_name'] = var_name
            
            # Build metadata
            combined_metadata = {
                'id': obj_id,
                **caller_info,
                **self.base_metadata,
            }
            
            # Add trace stack if available
            if self._trace_stack:
                combined_metadata['trace_id'] = self._trace_stack[-1]  # Current trace ID
                combined_metadata['trace_stack'] = self._trace_stack.copy()  # Full stack
            
            # Add call-specific metadata
            combined_metadata.update(metadata)
            
            # Write metadata
            with open(self.metadata_path, 'a', encoding='utf-8') as f:
                json.dump(combined_metadata, f)
                f.write('\n')
            
            # Store object
            with shelve.open(self.objects_path) as shelf:
                shelf[obj_id] = obj
                
        except Exception as e:
            raise RuntimeError(f"Error logging object: {str(e)}") from e
    
    def query(self, metadata_query: Dict[str, Any]) -> Iterator[Any]:
        """
        Query logged objects based on metadata.
        Special handling for trace_id to match anywhere in the trace stack.
        """
        def matches_query(metadata: Dict[str, Any], query: Dict[str, Any]) -> bool:
            for k, v in query.items():
                if k == 'trace_id':
                    direct_match = metadata.get('trace_id') == v
                    stack_match = v in metadata.get('trace_stack', []) 
                    if not (direct_match or stack_match):
                        return False
                else:
                    # Regular metadata matching
                    if metadata.get(k) != v:
                        return False
            return True

        try:
            with shelve.open(self.objects_path) as shelf:
                with open(self.metadata_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            metadata = json.loads(line.strip())
                            if matches_query(metadata, metadata_query):
                                obj_id = metadata['id']
                                
                                if shelf.get(obj_id, None) is not None:
                                    yield shelf.get(obj_id)
                        except json.JSONDecodeError:
                            continue
                            
        except Exception as e:
            raise RuntimeError(f"Error querying objects: {str(e)}") from e


    def query_with_metadata(self, metadata_query: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        """
        Query logged objects based on metadata and return both objects and their metadata.
        Special handling for trace_id to match anywhere in the trace stack.
        
        Args:
            metadata_query: Dictionary of metadata key-value pairs to match
            
        Returns:
            Iterator of dictionaries containing:
                - 'object': The stored object
                - 'metadata': The full metadata record for this object
        """
        def matches_query(metadata: Dict[str, Any], query: Dict[str, Any]) -> bool:
            for k, v in query.items():
                if k == 'trace_id':
                    direct_match = metadata.get('trace_id') == v
                    stack_match = v in metadata.get('trace_stack', []) 
                    if not (direct_match or stack_match):
                        return False
                else:
                    # Regular metadata matching
                    if metadata.get(k) != v:
                        return False
            return True

        try:
            with shelve.open(self.objects_path) as shelf:
                with open(self.metadata_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            metadata = json.loads(line.strip())
                            if matches_query(metadata, metadata_query):
                                obj_id = metadata['id']
                                
                                if shelf.get(obj_id, None) is not None:
                                    yield {
                                        'object': shelf[obj_id],
                                        'metadata': metadata
                                    }
                        except json.JSONDecodeError:
                            continue
                                
        except Exception as e:
            raise RuntimeError(f"Error querying objects: {str(e)}") from e



def load_experiment_logs(
    metadata_path: Union[str, Path],
    include_trace_stack: bool = True,  # Changed default to True since we're adding better handling
    additional_explode_columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Load experimental logs from a JSONL file into a pandas DataFrame with proper handling
    of trace IDs and nested structures.
    
    Args:
        metadata_path: Path to the JSONL metadata file
        include_trace_stack: Whether to keep and unroll the trace stack into separate columns
        additional_explode_columns: List of additional columns that contain JSON objects
            that should be exploded into separate columns
    
    Returns:
        pd.DataFrame: DataFrame containing the parsed logs with columns for:
            - Basic metadata (id, timestamp, function_name, etc.)
            - Trace information (trace_id, trace_stack.1, trace_stack.2, etc.)
            - Any additional metadata fields from the logs
    """
    # Read all lines from the JSONL file
    records = []
    with open(metadata_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                record = json.loads(line.strip())
                records.append(record)
            except json.JSONDecodeError:
                continue
    
    # Convert to DataFrame
    df = pd.DataFrame(records)
    
    # Convert id to datetime if it contains timestamp information
    if 'id' in df.columns:
        try:
            df['timestamp'] = pd.to_numeric(df['id']).apply(pd.Timestamp.fromtimestamp)
        except (ValueError, TypeError):
            pass
    
    # Handle trace information
    if 'trace_stack' in df.columns and include_trace_stack:
        # First, find the maximum depth of any trace stack
        max_depth = 0
        for stack in df['trace_stack'].dropna():
            if isinstance(stack, list):
                max_depth = max(max_depth, len(stack))
        
        # Create new columns for each level of the stack
        for depth in range(max_depth):
            col_name = f'trace_stack.{depth}'
            df[col_name] = None  # Initialize with None
            
            # Fill in values row by row
            for idx, stack in df['trace_stack'].items():
                try:
                    if pd.isna(stack) or not isinstance(stack, list):
                        continue
                except Exception:
                    continue
                if depth < len(stack):
                    df.at[idx, col_name] = stack[depth]
        
        # Drop the original trace_stack column
        df = df.drop('trace_stack', axis=1)
        
        # Add depth of call stack
        df['call_depth'] = df.filter(like='trace_stack').notna().sum(axis=1)
    
    # Explode additional JSON columns if specified
    if additional_explode_columns:
        for col in additional_explode_columns:
            if col in df.columns:
                # Try to parse the column as JSON if it's not already a dict
                if not isinstance(df[col].iloc[0], dict):
                    df[col] = df[col].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
                
                # Create new columns for each key in the JSON objects
                json_df = pd.json_normalize(df[col].dropna())
                
                # Add prefix to avoid column name conflicts
                json_df = json_df.add_prefix(f"{col}_")
                
                # Join with original DataFrame
                df = df.drop(col, axis=1).join(json_df)
    
    return df

def params_and_trace_ids_by_function(log_folder, metadata_df, function_to_get):
    shelf = shelve.open(str(_objects_store_path(log_folder)))
    metadata_params_df = metadata_df[(metadata_df["variable_name"] == "params") & (metadata_df["function_name"] == function_to_get)]
    params_list = [shelf.get(obj_id) for obj_id in metadata_params_df["id"].tolist()]
    trace_ids_list = metadata_params_df["trace_id"].tolist()
    assert len(params_list) == len(set(trace_ids_list))
    shelf.close()
    return params_list, trace_ids_list
