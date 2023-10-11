import app

def test_nombre_features_client():
    """Vérifie que la structure des données correspond à ce qu'attend le modèle en entrée."""
    model, clients = app.imports()
    assert clients.shape[1] == 757