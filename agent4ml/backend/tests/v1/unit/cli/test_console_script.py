import tomllib


def test_pyproject_exposes_agent4ml_console_script() -> None:
    with open("pyproject.toml", "rb") as file:
        pyproject = tomllib.load(file)

    assert pyproject["project"]["scripts"]["agent4ml"] == "agent4ml.backend.app.cli.main:main"
