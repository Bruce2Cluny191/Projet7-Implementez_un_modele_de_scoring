import app

def test_nombre_features_client():
    model, clients = app.imports()
    assert clients.shape[1] == 757