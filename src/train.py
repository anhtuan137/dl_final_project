from tensorflow import keras


def create_callbacks(path_checkpoint: str, patience: int = 20):
    es_callback = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0,
        patience=patience,
        restore_best_weights=True,
    )

    modelckpt_callback = keras.callbacks.ModelCheckpoint(
        monitor="val_loss",
        filepath=path_checkpoint,
        verbose=0,
        save_weights_only=True,
        save_best_only=True,
    )

    return [es_callback, modelckpt_callback]


def train_model(
    model,
    train_X,
    train_y,
    valid_X,
    valid_y,
    epochs: int,
    batch_size: int,
    callbacks=None,
):
    history = model.fit(
        train_X,
        train_y,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(valid_X, valid_y),
        callbacks=callbacks or [],
    )
    return history
