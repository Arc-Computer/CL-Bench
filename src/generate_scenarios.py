from pathlib import Path
from typing import Optional

from .crm_sandbox import MockCrmApi
from .entity_sampler import EntitySampler, SamplerConfig
from .scenario_generator import ScenarioGenerator
from .scenario_validator import ScenarioValidator
from .manifest_writer import ManifestWriter


def generate_scenarios(
    target_count: int = 1500,
    success_ratio: float = 0.6,
    output_dir: Optional[Path] = None,
    seed: Optional[int] = None,
) -> None:
    if output_dir is None:
        output_dir = Path("artifacts/generated_scenarios")

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Initializing CRM API...")
    api = MockCrmApi()

    print(f"Configuring entity sampler...")
    config = SamplerConfig(seed=seed)
    sampler = EntitySampler(api, config)

    print(f"Creating scenario generator...")
    generator = ScenarioGenerator(api, sampler)

    print(f"Generating {target_count} scenarios ({success_ratio:.0%} success, {1-success_ratio:.0%} failure)...")
    scenarios = generator.generate_batch(target_count, success_ratio)

    print(f"Validating scenarios...")
    validator = ScenarioValidator()
    valid_scenarios, errors = validator.validate_all(scenarios)

    if errors:
        print(f"⚠️  Found {len(errors)} validation errors:")
        for error in errors[:10]:
            print(f"  - {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")

    print(f"✓ {len(valid_scenarios)} valid scenarios")

    print(f"Writing outputs...")
    writer = ManifestWriter()

    scenarios_path = output_dir / "scenarios.jsonl"
    writer.write_jsonl(valid_scenarios, scenarios_path)
    print(f"  - Scenarios: {scenarios_path}")

    coverage_path = output_dir / "coverage_report.md"
    writer.write_coverage_report(valid_scenarios, coverage_path)
    print(f"  - Coverage report: {coverage_path}")

    stats = validator.get_coverage_stats(valid_scenarios)
    print(f"\n✓ Generation complete!")
    print(f"  Total: {stats['total_scenarios']}")
    print(f"  Success: {stats['success_scenarios']} ({stats.get('success_ratio', 0):.1%})")
    print(f"  Failure: {stats['failure_scenarios']} ({stats.get('failure_ratio', 0):.1%})")
    print(f"  Tasks covered: {len(stats['by_task'])}")
    print(f"  Intent categories: {len(stats['by_intent'])}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate CRM benchmark scenarios")
    parser.add_argument("--count", type=int, default=1500, help="Number of scenarios to generate")
    parser.add_argument("--success-ratio", type=float, default=0.6, help="Ratio of success scenarios (0.0-1.0)")
    parser.add_argument("--output-dir", type=str, help="Output directory path")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")

    args = parser.parse_args()

    output_path = Path(args.output_dir) if args.output_dir else None

    generate_scenarios(
        target_count=args.count,
        success_ratio=args.success_ratio,
        output_dir=output_path,
        seed=args.seed,
    )
