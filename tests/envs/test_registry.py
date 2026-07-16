from gymkhana.envs import get_environment

def test_registry_lookup():
    env_cls = get_environment("tool_use_singleturn")
    assert env_cls is not None
    assert get_environment("tool-use-singleturn") is env_cls
    instance = env_cls(config=None)
    assert instance.name == "tool_use_singleturn"

if __name__ == "__main__":
    test_registry_lookup()
    print("Registry lookup successful!")
