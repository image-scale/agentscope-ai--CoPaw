# Acceptance Criteria

## Task 1: Environment variable loading and path configuration

### Acceptance Criteria
- [ ] EnvVarLoader.get_bool("VAR", False) returns False when VAR is not set
- [ ] EnvVarLoader.get_bool("VAR", False) returns True when VAR is "true", "1", or "yes" (case-insensitive)
- [ ] EnvVarLoader.get_int("VAR", 10) returns 10 when VAR is not set
- [ ] EnvVarLoader.get_int("VAR", 10, min_value=0, max_value=100) clamps values to bounds
- [ ] EnvVarLoader.get_float("VAR", 1.0) returns 1.0 when VAR is not set
- [ ] EnvVarLoader.get_float with min_value/max_value clamps values correctly
- [ ] EnvVarLoader.get_float returns default when value is invalid
- [ ] EnvVarLoader.get_str("VAR", "default") returns "default" when VAR is not set
- [ ] WORKING_DIR is derived from COPAW_WORKING_DIR env var or defaults to ~/.copaw
- [ ] SECRET_DIR is derived from COPAW_SECRET_DIR env var or defaults to {WORKING_DIR}.secret
- [ ] CONFIG_FILE, JOBS_FILE, CHATS_FILE constants are defined with defaults
- [ ] LLM configuration constants (MAX_RETRIES, BACKOFF_BASE, etc.) have correct defaults
