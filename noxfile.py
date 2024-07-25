import os

import nox

package = "fingerling"
nox.options.sessions = ["tests"]
locations = "src", "tests", "noxfile.py"
os.environ["PDM_IGNORE_SAVED_PYTHON"] = "1"
python_versions = ["3.10", "3.11", "3.12"]


@nox.session(python=python_versions)
def tests(session: nox.session) -> None:
    session.run_always("maturin", "develop", external=True)
    session.run_always("pdm", "install", "-G", "test", external=True)
    session.run("pytest")
