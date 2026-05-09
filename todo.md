# Todo

## Plan
Implement the project in dependency order: start with configuration and constants, then build the provider abstraction layer, followed by concrete provider implementations, provider manager for multi-provider coordination, local model management, and finally the CLI interface. Each feature delivers complete testable functionality.

## Tasks
- [x] Task 1: Implement environment variable loading and path configuration (constant.py with EnvVarLoader, working directories, config paths)
- [x] Task 2: Implement LLM provider abstraction with model information and provider configuration models (provider.py with ModelInfo, ProviderInfo, Provider base class)
- [x] Task 3: Implement OpenAI-compatible provider that connects to OpenAI API, checks connectivity, and fetches available models (openai_provider.py)
- [x] Task 4: Implement Anthropic provider that connects to Anthropic API with connection checking and model management (anthropic_provider.py)
- [x] Task 5: Implement Ollama provider for local Ollama server connectivity and model discovery (ollama_provider.py)
- [>] Task 6: Implement provider manager that coordinates multiple providers, handles activation, and persists configuration to JSON files (provider_manager.py)
- [ ] Task 7: Implement download progress tracking system for model downloads with status tracking, cancellation support, and progress snapshots (download_manager.py)
- [ ] Task 8: Implement local model manager for downloading, listing, and managing locally stored LLM models from HuggingFace/ModelScope (model_manager.py)
- [ ] Task 9: Implement llama.cpp backend for running local models with server lifecycle management, binary installation, and model loading (llamacpp.py)
- [ ] Task 10: Implement unified local model manager facade that combines download manager, model manager, and llama.cpp backend (manager.py)
- [ ] Task 11: Implement Pydantic configuration models for agent config, channel configs, and runtime settings (config.py)
- [ ] Task 12: Implement CLI main entry point with lazy-loading subcommand support using Click framework (main.py)
- [ ] Task 13: Implement CLI init command for initializing workspace with configuration prompts and defaults (init_cmd.py)
- [ ] Task 14: Implement CLI providers/models command for listing and managing LLM providers (providers_cmd.py)
- [ ] Task 15: Implement workspace system for managing agent runtime environments with service coordination (workspace.py)
