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
- [x] ModelInfo stores model id, name, and optional multimodal capabilities flags
- [x] ProviderInfo stores provider id, name, base_url, api_key, and model lists
- [x] Provider base class defines abstract methods: check_connection, fetch_models, check_model_connection, get_chat_model_instance
- [x] Provider.has_model(model_id) returns True if model exists in models or extra_models
- [x] Provider.update_config(config) updates name, base_url, api_key, extra_models from dict
- [x] Provider.add_model() adds a model to extra_models and returns success tuple
- [x] Provider.add_model() returns (False, error) if model already exists
- [x] Provider.delete_model() removes model from extra_models
- [x] Provider.get_info() returns ProviderInfo with masked api_key when mock_secret=True

## Task 3: OpenAI-compatible provider

### Acceptance Criteria
- [ ] OpenAIProvider initializes with default OpenAI API base URL
- [ ] OpenAIProvider.check_connection() returns (True, "OK") on successful API connection
- [ ] OpenAIProvider.check_connection() returns (False, error_message) on connection failure
- [ ] OpenAIProvider.fetch_models() returns list of available models from API
- [ ] OpenAIProvider.check_model_connection(model_id) validates specific model is accessible
- [ ] OpenAIProvider includes built-in model definitions (GPT-4, GPT-4o, GPT-3.5-turbo, etc.)
- [ ] OpenAIProvider supports custom base_url for OpenAI-compatible APIs
- [ ] OpenAIProvider.get_chat_model_instance() returns configured model instance
