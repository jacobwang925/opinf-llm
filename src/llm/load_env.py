#!/usr/bin/env python3
"""
Load environment variables from .env file

Usage:
    from load_env import load_env
    load_env()
"""

import os
from pathlib import Path


def load_env(env_file=".env"):
    """
    Load environment variables from .env file.

    Args:
        env_file: Path to .env file (default: .env in current directory)
    """
    env_path = Path(env_file)
    if not env_path.exists():
        alt_path = Path(__file__).resolve().parent / env_file
        if alt_path.exists():
            env_path = alt_path
        else:
            print(f"⚠️  {env_file} not found. Create it from .env.example:")
            print(f"   cp .env.example .env")
            print(f"   # Then edit .env and add your API keys")
            return False

    # Read and parse .env file
    loaded_keys = []
    found_keys = []
    with open(env_path, 'r') as f:
        for line in f:
            line = line.strip()

            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue

            # Parse KEY=VALUE
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()

                # Only set if value is not empty and not already set
                if value:
                    found_keys.append(key)
                    if not os.environ.get(key):
                        os.environ[key] = value
                        loaded_keys.append(key)

    if loaded_keys:
        print(f"✓ Loaded API keys from {env_path}: {', '.join(loaded_keys)}")
        return True
    if found_keys:
        print(f"✓ API keys already set in environment (from {env_path})")
        return True
    print(f"⚠️  No API keys found in {env_path}. Make sure to fill in your keys.")
    return False


if __name__ == "__main__":
    # Test loading
    load_env()

    # Show which keys are set (without revealing values)
    api_keys = ['OPENAI_API_KEY', 'GEMINI_API_KEY', 'ANTHROPIC_API_KEY']
    print("\nAPI Key Status:")
    for key in api_keys:
        value = os.environ.get(key, '')
        if value and value != 'your-' + key.lower().replace('_', '-') + '-here':
            # Show first 10 chars
            print(f"  {key}: {'*' * min(10, len(value))}... ✓")
        else:
            print(f"  {key}: (not set)")
