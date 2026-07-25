"""Compatibility shim: real test lives under tests/motifs/."""
from dataset_clip_wrapper.tests.motifs.smoke_test_motif_agent import *  # noqa: F401,F403

if __name__ == "__main__":
    test_motif_agent()
    test_llm_motif_agent_with_mock_clients()
    print("motif agent smoke test passed")
