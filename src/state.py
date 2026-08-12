from collections import OrderedDict
from typing import Optional, Any, Dict

# Mutable server state — always access via `import state; state.xxx` (never `from state import xxx`)
# so that reassignments in one module are visible in all others.

current_model: Optional[Any] = None
current_model_name: Optional[str] = None

model_cache: "OrderedDict[str, Any]" = OrderedDict()
model_meta: Dict[str, Dict[str, Any]] = {}
failed_loads: Dict[str, float] = {}

mm_processor_cache: Dict[str, Any] = {}
