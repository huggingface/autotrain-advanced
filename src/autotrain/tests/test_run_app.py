from autotrain.cli.run_app import _uvicorn_command


def test_uvicorn_command_keeps_host_as_single_argument():
    host = "127.0.0.1; touch /tmp/autotrain-shell-injection"

    assert _uvicorn_command(host, 7860, 2) == [
        "uvicorn",
        "autotrain.app.app:app",
        "--host",
        host,
        "--port",
        "7860",
        "--workers",
        "2",
    ]
