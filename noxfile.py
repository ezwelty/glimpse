"""Nox sessions."""
import nox
from nox.sessions import Session

nox.options.sessions = "lint", "test"
locations = "src", "tests", "noxfile.py"


@nox.session(python=False)
def test(session: Session) -> None:
    """Test with pytest."""
    args = session.posargs or ["--doctest-modules", "src", "tests"]
    session.run("poetry", "install")
    session.run("poetry", "run", "pytest", *args)


@nox.session(python=False)
def lint(session: Session) -> None:
    """Lint with flake8."""
    args = session.posargs or locations
    session.run("poetry", "install", "--no-root")
    session.run("poetry", "run", "flake8", *args)


@nox.session(python=False)
def format(session: Session) -> None:
    """Format with black."""
    args = session.posargs or locations
    session.run("poetry", "install", "--no-root")
    session.run("poetry", "run", "black", *args)
