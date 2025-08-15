# agents/__init__.py

# Use the full path for all imports.
from agents.specialist_agent import SpecialistAgent
from agents.swarm import Swarm

# This is our catalog.
AVAILABLE_AGENTS = {
    'SpecialistAgent': SpecialistAgent,
}