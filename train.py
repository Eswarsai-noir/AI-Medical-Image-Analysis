from sklearn.model_selection import train_test_split

def train_model(model, X, y):
    X_train, X_val, y_train, y_val =train_test_split(X, y, test_size=0.2, random_state=42)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=5,
        batch_size=16
    )

    return history