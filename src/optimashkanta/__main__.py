"""Command-line interface."""
import click


@click.command()
@click.version_option()
def main() -> None:
    """OptiMashkanta."""


if __name__ == "__main__":
    main(prog_name="optimashkanta")  # pragma: no cover
