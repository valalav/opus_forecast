"""
Test Hypothesis Generator Agent
=================================
Tests for agents/hypothesis_generator.py
"""

import json
import tempfile
import shutil
from pathlib import Path
import pytest

# Import the agent
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from agents.hypothesis_generator import HypothesisGenerator, Hypothesis


class TestHypothesisGenerator:
    """Test suite for Hypothesis Generator Agent"""

    def test_hypothesis_creation(self):
        """Test that a hypothesis can be created"""
        hypothesis = Hypothesis(
            title="Test: USD lag 3 correlation",
            description="Test if USD with lag 3 improves MAE",
            rationale="Literature suggests 3-month lag",
            priority="medium",
        )
        assert hypothesis.title == "Test: USD lag 3 correlation"
        assert hypothesis.rationale == "Literature suggests 3-month lag"
        assert hypothesis.priority == "medium"

    def test_hypothesis_to_dict(self):
        """Test converting hypothesis to dictionary"""
        hypothesis = Hypothesis(
            title="Test Hypothesis",
            description="Test description",
            rationale="Test rationale",
            priority="low",
        )
        result = hypothesis.to_dict()
        assert result["title"] == "Test Hypothesis"
        assert result["description"] == "Test description"
        assert result["rationale"] == "Test rationale"
        assert result["priority"] == "low"

    def test_generator_initialization(self):
        """Test that generator can be initialized"""
        generator = HypothesisGenerator(
            prd_path=Path(__file__).parent.parent / "tasks" / "prd.json"
        )
        assert generator is not None
        assert generator.prd_path.exists()

    def test_generator_loads_prd(self):
        """Test that generator loads PRD correctly"""
        generator = HypothesisGenerator(
            prd_path=Path(__file__).parent.parent / "tasks" / "prd.json"
        )
        prd = generator.load_prd()
        assert "user_stories" in prd
        assert len(prd["user_stories"]) > 0

    def test_generate_hypothesis_basic(self):
        """Test basic hypothesis generation"""
        generator = HypothesisGenerator(
            prd_path=Path(__file__).parent.parent / "tasks" / "prd.json"
        )
        hypothesis = generator.generate_hypothesis()
        assert hypothesis is not None
        assert hypothesis.title.startswith("Hypothesis:")
        assert len(hypothesis.description) > 0
        assert len(hypothesis.rationale) > 0

    def test_generate_multiple_hypotheses(self):
        """Test generating multiple hypotheses"""
        generator = HypothesisGenerator(
            prd_path=Path(__file__).parent.parent / "tasks" / "prd.json"
        )
        hypotheses = generator.generate_hypotheses(count=3)
        assert len(hypotheses) == 3
        for h in hypotheses:
            assert isinstance(h, Hypothesis)

    def test_hypothesis_to_task_format(self):
        """Test converting hypothesis to PRD task format"""
        generator = HypothesisGenerator(
            prd_path=Path(__file__).parent.parent / "tasks" / "prd.json"
        )
        hypothesis = Hypothesis(
            title="Test: New feature X",
            description="Implement feature X",
            rationale="Feature X improves accuracy",
            priority="high",
        )
        task = generator.hypothesis_to_task(hypothesis)
        assert "title" in task
        assert "description" in task
        assert "acceptance_criteria" in task
        assert "status" in task
        assert task["status"] == "TODO"

    def test_append_task_to_prd(self):
        """Test appending a task to PRD (in-memory, no file write)"""
        # Create temporary copy of PRD
        with tempfile.TemporaryDirectory() as tmpdir:
            original_prd_path = Path(__file__).parent.parent / "tasks" / "prd.json"
            tmp_prd_path = Path(tmpdir) / "prd.json"
            shutil.copy(original_prd_path, tmp_prd_path)

            generator = HypothesisGenerator(prd_path=tmp_prd_path)
            original_count = len(generator.load_prd()["user_stories"])

            hypothesis = Hypothesis(
                title="Test: Add new correlation",
                description="Test adding new task",
                rationale="Testing",
                priority="low",
            )

            # Append task
            success = generator.append_task_to_prd(hypothesis)
            assert success is True

            # Verify task was added
            updated_prd = generator.load_prd()
            assert len(updated_prd["user_stories"]) == original_count + 1

            # Find the new task
            new_task = [
                t
                for t in updated_prd["user_stories"]
                if "Add new correlation" in t["title"]
            ]
            assert len(new_task) == 1

    def test_brainstorm_session(self):
        """Test a full brainstorm session"""
        with tempfile.TemporaryDirectory() as tmpdir:
            original_prd_path = Path(__file__).parent.parent / "tasks" / "prd.json"
            tmp_prd_path = Path(tmpdir) / "prd.json"
            shutil.copy(original_prd_path, tmp_prd_path)

            generator = HypothesisGenerator(prd_path=tmp_prd_path)
            original_count = len(generator.load_prd()["user_stories"])

            # Run brainstorm
            new_tasks = generator.brainstorm(count=2)
            assert len(new_tasks) == 2

            # Verify tasks were added
            updated_prd = generator.load_prd()
            assert len(updated_prd["user_stories"]) == original_count + 2

    def test_get_next_task_id(self):
        """Test getting next available task ID"""
        with tempfile.TemporaryDirectory() as tmpdir:
            original_prd_path = Path(__file__).parent.parent / "tasks" / "prd.json"
            tmp_prd_path = Path(tmpdir) / "prd.json"
            shutil.copy(original_prd_path, tmp_prd_path)

            generator = HypothesisGenerator(prd_path=tmp_prd_path)
            next_id = generator.get_next_task_id()
            assert next_id > 100  # We know there are tasks up to 100+

    def test_duplicate_task_detection(self):
        """Test that duplicate tasks are not added"""
        with tempfile.TemporaryDirectory() as tmpdir:
            original_prd_path = Path(__file__).parent.parent / "tasks" / "prd.json"
            tmp_prd_path = Path(tmpdir) / "prd.json"
            shutil.copy(original_prd_path, tmp_prd_path)

            generator = HypothesisGenerator(prd_path=tmp_prd_path)
            original_count = len(generator.load_prd()["user_stories"])

            # Add a task
            hypothesis = Hypothesis(
                title="Test: Unique task",
                description="Test description",
                rationale="Test",
                priority="low",
            )
            generator.append_task_to_prd(hypothesis)

            # Try to add the same task again (should not add duplicate)
            result = generator.append_task_to_prd(hypothesis)
            assert result is False

            updated_prd = generator.load_prd()
            assert (
                len(updated_prd["user_stories"]) == original_count + 1
            )  # Only one added


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
