# Acceptance Criteria

## Task 1: Environment variable loading and path configuration

### Acceptance Criteria
- [x] EnvVarLoader.get_bool("VAR", False) returns False when VAR is not set
- [x] EnvVarLoader.get_bool("VAR", False) returns True when VAR is "true", "1", or "yes" (case-insensitive)
- [x] EnvVarLoader.get_int("VAR", 10) returns 10 when VAR is not set
- [x] EnvVarLoader.get_int("VAR", 10, min_value=0, max_value=100) clamps values to bounds
- [x] EnvVarLoader.get_float("VAR", 1.0) returns 1.0 when VAR is not set
- [x] EnvVarLoader.get_float with min_value/max_value clamps values correctly
- [x] EnvVarLoader.get_float returns default when value is invalid
- [x] EnvVarLoader.get_str("VAR", "default") returns "default" when VAR is not set
- [x] WORKING_DIR is derived from COPAW_WORKING_DIR env var or defaults to ~/.copaw
- [x] SECRET_DIR is derived from COPAW_SECRET_DIR env var or defaults to {WORKING_DIR}.secret
- [x] CONFIG_FILE, JOBS_FILE, CHATS_FILE constants are defined with defaults
- [x] LLM configuration constants (MAX_RETRIES, BACKOFF_BASE, etc.) have correct defaults

## Task 2: LLM provider abstraction

### Acceptance Criteria
- [ ] ModelInfo stores model id, name, and optional multimodal capabilities flags
- [ ] ProviderInfo stores provider id, name, base_url, api_key, and model lists
- [ ] Provider base class defines abstract methods: check_connection, fetch_models, check_model_connection, get_chat_model_instance
- [ ] Provider.has_model(model_id) returns True if model exists in models or extra_models
- [ ] Provider.update_config(config) updates name, base_url, api_key, extra_models from dict
- [ ] Provider.add_model() adds a model to extra_models and returns success tuple
- [ ] Provider.add_model() returns (False, error) if model already exists
- [ ] Provider.delete_model() removes model from extra_models
- [ ] Provider.get_info() returns ProviderInfo with masked api_key when mock_secret=True
