"""Nox sessions."""
import tempfile
from typing import Any

import nox
from nox.sessions import Session

nox.options.sessions = "lint", "test"
locations = "src", "tests", "noxfile.py"


def install_with_constraints(session: Session, *args: str, **kwargs: Any) -> None:
    """
    Install packages constrained by Poetry's lock file.

    This function wraps :meth:`nox.sessions.Session.install`.
    It invokes `pip` to install packages inside of the session's virtualenv,
    pinned to the versions specified in `poetry.lock`.

    Arguments:
        session: Session to install packages into.
        args: Command-line arguments for `pip`.
        kwargs: Additional keyword arguments for :meth:`nox.sessions.Session.install`.
    """
    with tempfile.NamedTemporaryFile() as requirements:
        session.run(
            "poetry",
            "export",
            "--format=requirements.txt",
            "--without-hashes",
            f"--output={requirements.name}",
            external=True,
        )
        session.install(f"--constraint={requirements.name}", *args, **kwargs)


@nox.session(python=["3.6", "3.7", "3.8"])
def test(session: Session) -> None:
    """Test with pytest."""
    args = session.posargs or ["--cov", "--doctest-modules", "src", "tests"]
    session.run("poetry", "install", "--no-dev", external=True)
    install_with_constraints(session, "coverage[toml]", "pytest", "pytest-cov")
    session.run("pytest", *args)


@nox.session(python=["3.6", "3.7", "3.8"])
def lint(session: Session) -> None:
    """Lint with flake8."""
    args = session.posargs or locations
    install_with_constraints(
        session,
        "flake8",
        "flake8-annotations",
        "flake8-black",
        "flake8-docstrings",
        "flake8-import-order",
    )
    session.run("flake8", *args)


@nox.session(python="3.8")
def format(session: Session) -> None:
    """Format with black."""
    args = session.posargs or locations
    install_with_constraints(session, "black")
    session.run("black", *args)
