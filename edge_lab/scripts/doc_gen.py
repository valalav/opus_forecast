#!/usr/bin/env python3
"""
Documentation Generator for Sirena Models
==========================================
Extracts docstrings and __init__ parameters from all Forecaster classes.
"""

import inspect
import ast
from pathlib import Path
from typing import Dict, List, Any, Optional
import importlib


def extract_init_params_from_source(
    source_code: str, class_name: str
) -> List[Dict[str, Any]]:
    """Extract __init__ parameters from source code using AST."""
    params = []

    try:
        tree = ast.parse(source_code)

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                        # Extract parameters (skip 'self')
                        args = item.args
                        all_args = args.args + args.kwonlyargs

                        for i, arg in enumerate(all_args):
                            if arg.arg == "self":
                                continue

                            param_info = {
                                "name": arg.arg,
                                "type": "Any",
                                "default": None,
                            }

                            # Try to get default value
                            defaults_offset = len(all_args) - len(args.defaults)
                            default_idx = i - defaults_offset

                            if 0 <= default_idx < len(args.defaults):
                                default_node = args.defaults[default_idx]
                                param_info["default"] = (
                                    ast.unparse(default_node)
                                    if hasattr(ast, "unparse")
                                    else "..."
                                )

                            # Try to get type annotation
                            if arg.annotation:
                                param_info["type"] = (
                                    ast.unparse(arg.annotation)
                                    if hasattr(ast, "unparse")
                                    else str(arg.annotation)
                                )

                            params.append(param_info)

                        break
                break
    except Exception:
        pass

    return params


def extract_init_params_from_signature(cls) -> List[Dict[str, Any]]:
    """Extract __init__ parameters from class signature."""
    params = []

    try:
        init_method = cls.__init__
        sig = inspect.signature(init_method)

        for name, param in sig.parameters.items():
            if name == "self":
                continue

            param_info = {"name": name, "type": "Any", "default": None}

            if param.annotation != inspect.Parameter.empty:
                param_info["type"] = str(param.annotation)

            if param.default != inspect.Parameter.empty:
                param_info["default"] = str(param.default)

            params.append(param_info)
    except Exception:
        pass

    return params


def discover_models(
    models_dir: Path, additional_models_dir: Optional[Path] = None
) -> List[tuple]:
    """Discover all Forecaster classes in models directory using direct import."""
    classes_dict = {}

    # Add sirena to path
    import sys

    sirena_root = models_dir.parent.parent
    if str(sirena_root) not in sys.path:
        sys.path.insert(0, str(sirena_root))

    # Add edge_lab to path for additional models
    if additional_models_dir and additional_models_dir.exists():
        edge_lab_root = additional_models_dir.parent
        if str(edge_lab_root) not in sys.path:
            sys.path.insert(0, str(edge_lab_root))

    # Import sirena.models package (main)
    models_package = importlib.import_module("sirena.models")

    # Get all module names
    module_names = [
        f.name[:-3]  # Remove .py extension
        for f in sorted(models_dir.glob("*.py"))
        if f.name not in ["__init__.py", "base.py", "registry.py", "exog_loader.py"]
    ]

    print(f"Found {len(module_names)} module files to process")

    for module_name in module_names:
        try:
            # Import module from sirena.models
            full_module_name = f"sirena.models.{module_name}"
            module = importlib.import_module(full_module_name)

            # Find all Forecaster and Model classes in this module
            for name, obj in inspect.getmembers(module):
                if (
                    inspect.isclass(obj)
                    and ("Forecaster" in name or "Model" in name)
                    and name != "BaseForecaster"
                    and name != "ModelRegistry"
                ):
                    # Get source code for AST parsing
                    source_code = inspect.getsource(obj)

                    # Get docstring
                    docstring = inspect.getdoc(obj) or "No docstring available."

                    # Extract parameters
                    params = extract_init_params_from_source(source_code, name)
                    if not params:
                        params = extract_init_params_from_signature(obj)

                    # Get file and line info
                    try:
                        source_file = inspect.getsourcefile(obj)
                        lines, start_line = inspect.getsourcelines(obj)
                        end_line = start_line + len(lines) - 1
                    except Exception:
                        source_file = str(models_dir / f"{module_name}.py")
                        start_line = end_line = 0

                    # Deduplicate by class name (keep first found)
                    if name not in classes_dict:
                        classes_dict[name] = (
                            name,
                            obj,
                            docstring,
                            params,
                            source_file,
                            start_line,
                            end_line,
                        )
                        print(f"  Found: {name}")

        except Exception as e:
            print(f"  Error processing {module_name}: {e}")

    # Process additional models directory (edge_lab sirena.models)
    if additional_models_dir and additional_models_dir.exists():
        print(f"\nProcessing additional models directory: {additional_models_dir}")
        module_names_addl = [
            f.name[:-3]
            for f in sorted(additional_models_dir.glob("*.py"))
            if f.name not in ["__init__.py", "base.py", "registry.py"]
        ]

        for module_name in module_names_addl:
            try:
                full_module_name = f"sirena.models.{module_name}"
                module = importlib.import_module(full_module_name)

                for name, obj in inspect.getmembers(module):
                    if (
                        inspect.isclass(obj)
                        and "Forecaster" in name
                        and name != "BaseForecaster"
                    ):
                        source_code = inspect.getsource(obj)
                        docstring = inspect.getdoc(obj) or "No docstring available."

                        params = extract_init_params_from_source(source_code, name)
                        if not params:
                            params = extract_init_params_from_signature(obj)

                        try:
                            source_file = inspect.getsourcefile(obj)
                            lines, start_line = inspect.getsourcelines(obj)
                            end_line = start_line + len(lines) - 1
                        except Exception:
                            source_file = str(
                                additional_models_dir / f"{module_name}.py"
                            )
                            start_line = end_line = 0

                        if name not in classes_dict:
                            classes_dict[name] = (
                                name,
                                obj,
                                docstring,
                                params,
                                source_file,
                                start_line,
                                end_line,
                            )
                            print(f"  Found (additional): {name}")

            except Exception as e:
                print(f"  Error processing additional {module_name}: {e}")

    return list(classes_dict.values())


def generate_markdown(classes: List[tuple], output_path: Path):
    """Generate markdown documentation from extracted classes."""
    lines = [
        "# Sirena Models - Auto-Generated Documentation",
        "",
        f"Generated by: scripts/doc_gen.py",
        f"Total Models: {len(classes)}",
        "",
        "---",
        "",
    ]

    for class_name, cls, docstring, params, source_file, start_line, end_line in sorted(
        classes, key=lambda x: x[0]
    ):
        lines.append(f"## {class_name}")
        lines.append("")
        lines.append(f"**Source:** `{source_file}:{start_line}-{end_line}`")
        lines.append("")
        lines.append("### Description")
        lines.append("")
        lines.append(f"```")
        lines.append(docstring.strip())
        lines.append("```")
        lines.append("")

        if params:
            lines.append("### Parameters")
            lines.append("")
            lines.append("| Parameter | Type | Default | Description |")
            lines.append("|-----------|------|---------|-------------|")

            for param in params:
                param_name = param["name"]
                param_type = param.get("type", "Any")
                param_default = param.get("default", "None")
                if param_default == "None":
                    param_default = "required"
                lines.append(f"| `{param_name}` | {param_type} | {param_default} | |")

            lines.append("")

        lines.append("---")
        lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))


def main():
    """Main function."""
    # Main sirena models directory
    models_dir = Path(__file__).parent.parent.parent / "sirena" / "models"
    # Additional edge_lab models directory
    edge_lab_models_dir = Path(__file__).parent.parent / "sirena" / "models"
    output_path = Path(__file__).parent.parent / "docs" / "MODELS_AUTO.md"

    if not models_dir.exists():
        print(f"Error: Models directory not found at {models_dir}")
        return 1

    print("Discovering models...")
    # Only process main sirena.models directory (not edge_lab duplicate)
    classes = discover_models(models_dir, None)

    if not classes:
        print("Warning: No Forecaster classes found!")
        return 1

    print(f"\nFound {len(classes)} model classes:")
    for name, *_ in sorted(classes):
        print(f"  - {name}")

    print("\nGenerating documentation...")
    generate_markdown(classes, output_path)

    print(f"Documentation saved to: {output_path}")
    return 0


if __name__ == "__main__":
    exit(main())
