"""
Hypothesis Generator Agent
==========================
Autonomous agent that brainstorm new correlation theories and appends tasks to PRD.

This agent analyzes economic data and generates hypotheses about potential
correlations that could improve forecasting models.
"""

import json
import random
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
import hashlib


@dataclass
class Hypothesis:
    """Represents a hypothesis about economic correlations"""

    title: str
    description: str
    rationale: str
    priority: str = "medium"
    confidence: float = 0.5
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert hypothesis to dictionary"""
        return {
            "title": self.title,
            "description": self.description,
            "rationale": self.rationale,
            "priority": self.priority,
            "confidence": self.confidence,
            "created_at": self.created_at,
        }


class HypothesisGenerator:
    """
    Autonomous agent for generating hypotheses about economic correlations.

    Uses pattern recognition and economic theory to propose new features,
    model architectures, or research directions for improving forecasts.
    """

    # Economic variables available in data
    VARIABLES = {
        "usd_nom_i": {"name": "USD/RUB exchange rate", "type": "exchange"},
        "Ki_i": {"name": "Key rate (ЦБ РФ)", "type": "monetary"},
        "Ruonia": {"name": "Ruonia rate", "type": "monetary"},
        "brent": {"name": "Brent oil price", "type": "commodity"},
        "mom": {"name": "Inflation (MoM)", "type": "target"},
        "Prod": {"name": "Food prices", "type": "component"},
        "Nonprod": {"name": "Non-food prices", "type": "component"},
        "Serv": {"name": "Services prices", "type": "component"},
    }

    # Lag ranges to test
    LAGS = [1, 2, 3, 6, 12]

    # Hypothesis templates
    TEMPLATES = [
        ("lag", "Test if {var} with lag {lag} improves MAE"),
        ("volatility", "Test {var} volatility (rolling std) as feature"),
        ("momentum", "Test {var} momentum (first difference) as feature"),
        ("interaction", "Test interaction between {var1} and {var2}"),
        ("threshold", "Test {var} threshold effect (above/below X)"),
        ("nonlinear", "Test nonlinear transformation of {var} (log, square)"),
    ]

    def __init__(self, prd_path: Path):
        """
        Initialize hypothesis generator.

        Args:
            prd_path: Path to PRD JSON file
        """
        self.prd_path = prd_path
        self.generated_hypotheses: List[str] = []

    def load_prd(self) -> Dict[str, Any]:
        """Load PRD from JSON file"""
        with open(self.prd_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_prd(self, prd: Dict[str, Any]) -> bool:
        """
        Save PRD to JSON file (atomically).

        Args:
            prd: PRD dictionary to save

        Returns:
            True if successful, False otherwise
        """
        try:
            # Write to temporary file first
            tmp_path = self.prd_path.with_suffix(".tmp")
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(prd, f, indent=4, ensure_ascii=False)

            # Atomic rename
            tmp_path.replace(self.prd_path)
            return True
        except Exception as e:
            print(f"Error saving PRD: {e}")
            return False

    def generate_hypothesis(self, topic: Optional[str] = None) -> Hypothesis:
        """
        Generate a single hypothesis.

        Args:
            topic: Optional topic to focus on (e.g., "USD", "Brent")

        Returns:
            Hypothesis object
        """
        # Select template and variables
        template_type, template = random.choice(self.TEMPLATES)
        variables = list(self.VARIABLES.keys())

        if template_type == "lag":
            var = random.choice(variables)
            lag = random.choice(self.LAGS)
            title = f"Hypothesis: {self.VARIABLES[var]['name']} lag {lag}"
            description = template.format(var=var, lag=lag)
            rationale = f"Economic literature suggests {lag}-month lag for {self.VARIABLES[var]['name']}"

        elif template_type == "volatility":
            var = random.choice(variables)
            title = f"Hypothesis: {self.VARIABLES[var]['name']} volatility"
            description = template.format(var=var)
            rationale = (
                f"Volatility captures uncertainty in {self.VARIABLES[var]['name']}"
            )

        elif template_type == "momentum":
            var = random.choice(variables)
            title = f"Hypothesis: {self.VARIABLES[var]['name']} momentum"
            description = template.format(var=var)
            rationale = (
                f"Momentum captures direction changes in {self.VARIABLES[var]['name']}"
            )

        elif template_type == "interaction":
            var1, var2 = random.sample(variables, 2)
            title = f"Hypothesis: {self.VARIABLES[var1]['name']} × {self.VARIABLES[var2]['name']}"
            description = template.format(var1=var1, var2=var2)
            rationale = f"Interaction effect between {self.VARIABLES[var1]['name']} and {self.VARIABLES[var2]['name']}"

        elif template_type == "threshold":
            var = random.choice(variables)
            threshold = random.choice([0.5, 1.0, 2.0])
            title = f"Hypothesis: {self.VARIABLES[var]['name']} threshold {threshold}"
            description = template.format(var=var)
            rationale = f"Nonlinear effect when {self.VARIABLES[var]['name']} exceeds {threshold}"

        elif template_type == "nonlinear":
            var = random.choice(variables)
            transform = random.choice(["log", "square", "sqrt"])
            title = f"Hypothesis: {transform}({self.VARIABLES[var]['name']})"
            description = template.format(var=var)
            rationale = f"Nonlinear transformation may capture {self.VARIABLES[var]['name']} dynamics"

        else:
            # Default hypothesis
            title = "Hypothesis: New feature set"
            description = "Test new combination of features"
            rationale = "Exploratory analysis"

        return Hypothesis(
            title=title,
            description=description,
            rationale=rationale,
            priority=random.choice(["high", "medium", "low"]),
            confidence=random.uniform(0.3, 0.8),
        )

    def generate_hypotheses(self, count: int = 5) -> List[Hypothesis]:
        """
        Generate multiple hypotheses.

        Args:
            count: Number of hypotheses to generate

        Returns:
            List of Hypothesis objects
        """
        return [self.generate_hypothesis() for _ in range(count)]

    def get_next_task_id(self) -> int:
        """Get the next available task ID from PRD"""
        prd = self.load_prd()
        existing_ids = [task.get("id", 0) for task in prd.get("user_stories", [])]
        return max(existing_ids) + 1 if existing_ids else 1

    def hypothesis_to_task(self, hypothesis: Hypothesis) -> Dict[str, Any]:
        """
        Convert hypothesis to PRD task format.

        Args:
            hypothesis: Hypothesis object

        Returns:
            Task dictionary in PRD format
        """
        task_id = self.get_next_task_id()

        return {
            "id": task_id,
            "title": hypothesis.title,
            "description": hypothesis.description,
            "acceptance_criteria": [
                "MAE improved or validated not useful",
                f"Hypothesis rationale: {hypothesis.rationale}",
            ],
            "status": "TODO",
            "priority": hypothesis.priority,
            "meta": {
                "rationale": hypothesis.rationale,
                "confidence": hypothesis.confidence,
                "created_at": hypothesis.created_at,
                "type": "hypothesis",
            },
        }

    def is_duplicate(self, hypothesis: Hypothesis) -> bool:
        """
        Check if hypothesis already exists in PRD.

        Args:
            hypothesis: Hypothesis to check

        Returns:
            True if duplicate exists, False otherwise
        """
        prd = self.load_prd()
        for task in prd.get("user_stories", []):
            if task.get("title") == hypothesis.title:
                return True
        return False

    def append_task_to_prd(self, hypothesis: Hypothesis) -> bool:
        """
        Append a hypothesis as a task to PRD.

        Args:
            hypothesis: Hypothesis to append

        Returns:
            True if successful, False if duplicate or error
        """
        # Check for duplicates
        if self.is_duplicate(hypothesis):
            return False

        prd = self.load_prd()

        # Convert to task
        task = self.hypothesis_to_task(hypothesis)

        # Append to user_stories
        prd["user_stories"].append(task)

        # Save
        return self.save_prd(prd)

    def brainstorm(self, count: int = 5) -> List[Dict[str, Any]]:
        """
        Run a brainstorm session to generate and append hypotheses.

        Args:
            count: Number of hypotheses to generate

        Returns:
            List of added tasks
        """
        added_tasks = []

        # Generate hypotheses
        hypotheses = self.generate_hypotheses(count=count)

        # Append to PRD
        for hypothesis in hypotheses:
            if self.append_task_to_prd(hypothesis):
                prd = self.load_prd()
                new_task = [
                    t for t in prd["user_stories"] if t.get("title") == hypothesis.title
                ][0]
                added_tasks.append(new_task)

        return added_tasks


def main():
    """Run hypothesis generator standalone"""
    prd_path = Path(__file__).parent.parent / "tasks" / "prd.json"
    generator = HypothesisGenerator(prd_path=prd_path)

    print("🧬 Hypothesis Generator Agent")
    print("=" * 50)
    print(f"PRD Path: {prd_path}")
    print()

    # Generate hypotheses
    print("Generating hypotheses...")
    hypotheses = generator.generate_hypotheses(count=3)

    for i, h in enumerate(hypotheses, 1):
        print(f"\n{i}. {h.title}")
        print(f"   Description: {h.description}")
        print(f"   Rationale: {h.rationale}")
        print(f"   Priority: {h.priority}, Confidence: {h.confidence:.2f}")

    print("\n" + "=" * 50)
    print("To append to PRD, call: generator.brainstorm(count=N)")


if __name__ == "__main__":
    main()
