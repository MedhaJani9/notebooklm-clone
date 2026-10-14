def test_ui_builds_and_handlers_wired(service):
    import app
    demo = app.build_app(service)
    assert demo is not None and len(demo.fns) > 10
