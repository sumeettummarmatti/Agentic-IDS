import sys
import os
import asyncio
import logging
import importlib.util
from typing import List, Dict, Any, Optional

# Import our local LLM client
# Fix path for local import if running from root
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.council.llm_council_wrapper import LLMClient

logger = logging.getLogger(__name__)

# --- HELPER TO LOAD MODULE FROM PATH ---
def load_module_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Path to llm-council backend
backend_path = os.path.join(current_dir, "llm-council", "backend")

# --- MONKEY PATCHING ---
# We need to intercept calls to OpenRouter and redirect them to our models

# 1. Initialize our client
# We will use the same client instance we use elsewhere
_local_client = LLMClient()

# 2. Define the replacement function for query_model
async def _patched_query_model(model: str, messages: List[Dict[str, str]], timeout: float = 120.0) -> Optional[Dict[str, Any]]:
    """
    Patched version of openrouter.query_model that uses our local LLMClient.
    Maps model names from the config to our available models.
    """
    # Map high-end models to our best available (Groq 70b)
    # Map fast models to our fast available (Groq 8b or Ollama)
    
    # Simple mapping logic
    target_provider = "groq"
    target_model_name = "llama-3.3-70b-versatile" # Default to strongest
    
    provider = os.getenv('LLM_PROVIDER', 'groq').lower()
    
    if provider == 'hf':
        target_provider = "hf"
        target_model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
        if "gemini" in model or "gpt" in model or "claude" in model:
             target_model_name = "mistralai/Mistral-7B-Instruct-v0.3"
    elif provider == 'ollama':
        target_provider = "ollama"
        target_model_name = "llama3.1:8b"
        if "gemini" in model or "gpt" in model or "claude" in model:
            target_model_name = "qwen2.5:8b"
    else:
        # Default Groq logic
        target_provider = "groq"
        if "gemini" in model or "gpt" in model:
            target_model_name = "llama-3.3-70b-versatile"
        elif "claude" in model:
            target_model_name = "llama-3.3-70b-versatile"
        elif "flash" in model or "fast" in model:
            target_model_name = "llama-3.1-8b-instant"
        
    # Construct prompt from messages
    # Messages are [{'role': 'user', 'content': '...'}, ...]
    # We can concatenate them or just take the last one if it's a simple query
    # The Karpathy council code sends full conversation history sometimes
    
    full_prompt = ""
    for msg in messages:
        role = msg['role'].upper()
        content = msg['content']
        full_prompt += f"\n{role}: {content}"
        
    full_prompt += "\nASSISTANT: "
    
    try:
        # Use our client
        # We need to map provider/model format "groq:modelname"
        model_id = f"{target_provider}:{target_model_name}"
        
        # Call Generate (synchronous in our client, but we wrap in async if needed?)
        # Actually our client uses invoke_model which is synchronous blocking IO (requests/groq library)
        # But this function must be async.
        
        # We run it in a thread to not block the event loop
        loop = asyncio.get_event_loop()
        response_text = await loop.run_in_executor(
            None, 
            lambda: _local_client.invoke_model(model_id, full_prompt)
        )
        
        # Format explicitly as per OpenRouter response expectation
        return {
            'content': response_text,
            'reasoning_details': None
        }
        
    except Exception as e:
        logger.error(f"Error in patched query_model: {e}")
        return None

# 3. Define parallel query function
async def _patched_query_models_parallel(models: List[str], messages: List[Dict[str, str]]) -> Dict[str, Optional[Dict[str, Any]]]:
    import asyncio
    tasks = [_patched_query_model(model, messages) for model in models]
    responses = await asyncio.gather(*tasks)
    return {model: response for model, response in zip(models, responses)}

# --- ROBUST LOADING STRATEGY ---
try:
    # 1. Ensure backend is a package
    if backend_path not in sys.path:
        sys.path.append(backend_path)
    # Also add the parent (llm-council) so "import backend" works
    llm_council_root = os.path.join(current_dir, "llm-council")
    if llm_council_root not in sys.path:
        sys.path.append(llm_council_root)

    # 2. Patch sys.modules to prevent relative import errors
    # We load config and openrouter manually first
    from backend import config as council_config
    from backend import openrouter 
    
    # Apply monkey patches to the loaded modules
    openrouter.query_model = _patched_query_model
    openrouter.query_models_parallel = _patched_query_models_parallel
    
    # Configure Persona Models
    council_config.COUNCIL_MODELS = [
        "Security-Expert-GPT4"
        "Network-Engineer-Claude",
        "Threat-Hunter-Gemini" 
    ]
    council_config.CHAIRMAN_MODEL = "Chairman-GPT4"
    
    # 3. Import Council
    # Since we are importing from 'backend', the relative imports inside backend/council.py 
    # (e.g. 'from .openrouter import ...') should work fine IF 'backend' is recognized as a package.
    from backend import council
    run_full_council = council.run_full_council

    logger.info("✓ Successfully patched Karpathy LLM Council")
    
except ImportError as e:
    logger.error(f"Failed to import llm-council modules. Error: {e}")
    # DETAILED DEBUGGING
    import traceback
    logger.error(traceback.format_exc())
    run_full_council = None

# --- PUBLIC API ---

async def find_solutions(council_analysis_text: str) -> str:
    """
    Uses the Karpathy Council to brainstorm solutions based on the analysis.
    """
    if not run_full_council:
        return "Error: Karpathy Council not loaded."
        
    query = f"""
    Based on the following Threat Analysis, propose 3 concrete technical solutions to mitigate this attack.
    Rank them by feasibility and impact.
    
    THREAT ANALYSIS:
    {council_analysis_text}
    """
    
    print("\n[Karpathy Council] Brainstorming solutions...")
    
    # Run the 3-stage process
    stage1, stage2, stage3, meta = await run_full_council(query)
    
    final_response = stage3.get('response', 'No consensus reached.')
    return final_response
