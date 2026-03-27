"""
main.py - AStats Orchestrator
Minimal human-in-the-loop workflow for agentic statistical analysis.

Usage:
    python main.py
    python main.py --csv sample_data/iris.csv --goal "perform EDA"
"""
import os
import sys
import argparse
from dotenv import load_dotenv

load_dotenv()

from agents import (
    data_understanding_agent,
    data_structure_agent,
    statistical_constraint_agent,
    planning_agent
)
from utils.file_utils import read_markdown, ensure_output_dir

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
SEPARATOR = "=" * 70


def _print_banner():
    print(f"""
{SEPARATOR}
   █████╗ ███████╗████████╗ █████╗ ████████╗███████╗
  ██╔══██╗██╔════╝╚══██╔══╝██╔══██╗╚══██╔══╝██╔════╝
  ███████║███████╗   ██║   ███████║   ██║   ███████╗
  ██╔══██║╚════██║   ██║   ██╔══██║   ██║   ╚════██║
  ██║  ██║███████║   ██║   ██║  ██║   ██║   ███████║
  ╚═╝  ╚═╝╚══════╝   ╚═╝   ╚═╝  ╚═╝   ╚═╝   ╚══════╝
  Agentic Statistical Analysis System  |  Prototype v0.1
{SEPARATOR}
""")


def _prompt_inputs(args) -> tuple[str, str]:
    """Get CSV path and analysis goal from CLI args or interactive prompts."""
    if args.csv:
        csv_path = args.csv
    else:
        csv_path = input("Enter path to your CSV dataset: ").strip()

    if not os.path.exists(csv_path):
        print(f"[ERROR] File not found: {csv_path}")
        sys.exit(1)

    if args.goal:
        user_goal = args.goal
    else:
        user_goal = input("Enter your analysis goal (e.g., 'perform EDA and apply regression'): ").strip()

    if not user_goal:
        user_goal = "Perform exploratory data analysis"

    return csv_path, user_goal


def _display_plan(plan_path: str):
    """Print the current analysis plan to the terminal."""
    plan_text = read_markdown(plan_path)
    print(f"\n{SEPARATOR}")
    print("CURRENT ANALYSIS PLAN")
    print(SEPARATOR)
    print(plan_text)
    print(SEPARATOR)
    return plan_text


def _human_in_the_loop(plan_path: str):
    """
    Human review loop:
    - User can approve the plan, provide feedback, or quit.
    - If feedback is given, the planning agent refines the plan.
    - Loop continues until approved or quit.
    """
    iteration = 0
    while True:
        plan_text = _display_plan(plan_path)
        iteration += 1

        print(f"\n[Review Round {iteration}]")
        print("Options:")
        print("  approve  -> Accept the plan and finish")
        print("  quit     -> Exit without approving")
        print("  <text>   -> Provide feedback to refine the plan\n")

        user_input = input("Your choice or feedback: ").strip()

        if not user_input:
            print("(No input given, please try again.)")
            continue

        if user_input.lower() in ("approve", "a", "yes", "ok", "done"):
            print(f"""
{SEPARATOR}
Plan approved! Outputs generated in outputs/:
    - dataset_profile.md   : Dataset profiling report
    - analysis_plan.md     : Your finalised analysis plan
    - analysis_notebook.ipynb : Starter Jupyter notebook

Open analysis_notebook.ipynb in Jupyter to begin your analysis.
{SEPARATOR}
""")
            break

        elif user_input.lower() in ("quit", "q", "exit"):
            print("\nExiting AStats. Outputs saved to the outputs/ directory.")
            sys.exit(0)

        else:
            print("\nSending feedback to the Planning Agent...")
            planning_agent.refine_plan(plan_text, user_input, OUTPUT_DIR)
            print("Plan updated. Displaying revised plan...\n")


def main():
    parser = argparse.ArgumentParser(description="AStats - Agentic Statistical Analysis System")
    parser.add_argument("--csv",  type=str, help="Path to the CSV dataset")
    parser.add_argument("--goal", type=str, help="Analysis goal (quoted string)")
    args = parser.parse_args()

    _print_banner()
    ensure_output_dir(OUTPUT_DIR)

    # Step 1: Get inputs
    csv_path, user_goal = _prompt_inputs(args)
    print(f"\nDataset : {csv_path}")
    print(f"Goal    : {user_goal}\n")

    # Step 2: Data Understanding Agent
    print(f"{SEPARATOR}")
    print("[Agent 1] Data Understanding Agent - starting...")
    print(SEPARATOR)
    profile_path = data_understanding_agent.run(csv_path, OUTPUT_DIR)

    # Step 3: Data Structure Inference Module
    print(f"\n{SEPARATOR}")
    print("[Agent 2] Data Structure Agent - inferring data generation process...")
    print(SEPARATOR)
    data_structure_path = data_structure_agent.run(csv_path, OUTPUT_DIR)

    # Step 4: Statistical Constraint Engine
    print(f"\n{SEPARATOR}")
    print("[Agent 3] Statistical Constraint Engine - generating constraints...")
    print(SEPARATOR)
    profile_text = read_markdown(profile_path)
    with open(data_structure_path, "r", encoding="utf-8") as f:
        data_structure_text = f.read()
    constraints_path = statistical_constraint_agent.run(profile_text, data_structure_text, OUTPUT_DIR)

    # Step 5: Planning Agent - Generate Plan
    print(f"\n{SEPARATOR}")
    print("[Agent 4] Planning Agent - generating constraint-aware analysis plan...")
    print(SEPARATOR)
    with open(constraints_path, "r", encoding="utf-8") as f:
        constraints_text = f.read()

    plan_path = planning_agent.generate_plan(profile_text, data_structure_text, constraints_text, user_goal, OUTPUT_DIR)

    # Step 6 & 7: Human-in-the-loop review + refinement
    print(f"\n{SEPARATOR}")
    print("[Human-in-the-Loop] Please review the generated analysis plan.")
    print(SEPARATOR)
    _human_in_the_loop(plan_path)


if __name__ == "__main__":
    main()
