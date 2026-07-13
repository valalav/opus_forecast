import sys
from pathlib import Path

# Add the directory to path
sys.path.append("/home/valalav/ito/ralph_universal")

try:
    print("Checking imports...")
    import config
    print("✅ config imported")
    from core.state import StateManager
    print("✅ core.state imported")
    from core.agent_wrapper import AgentWrapper
    print("✅ core.agent_wrapper imported")
    import worker
    print("✅ worker imported")
    import critic
    print("✅ critic imported")
    import orchestrator
    print("✅ orchestrator imported")
    
    print("\nChecking state management...")
    state = StateManager()
    prd = state.read_prd()
    print(f"✅ StateManager initialized. PRD stories: {len(prd.get('user_stories', []))}")
    
    print("\n🎉 Verification Successful! Ralph Universal is ready.")
    
except Exception as e:
    print(f"\n❌ Verification Failed: {e}")
    sys.exit(1)
