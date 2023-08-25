import nox


@nox.session
@nox.parametrize("x64", [True, False])
def tests(session, x64):
    session.install(".[test]")
    args = session.posargs
    if not args:
        args = ["-v"]
    if x64:
        env = {"JAX_ENABLE_X64": "1"}
    else:
        env = {}
    session.run("pytest", *args, env=env)


@nox.session
def doctest(session):
    session.install(".")
    session.run("python", "-m", "doctest", "-v", "README.md")
