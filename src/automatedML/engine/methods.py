import keras
import numpy as np
from sklearn.model_selection import KFold


def train_model(model: keras.Model,
                X_train, y_train,
                loss_type, training_optimizer, metrics_type,
                epochs, shuffle, validation_data, batch_size, callbacks):

    model.compile(
        loss=loss_type,
        optimizer=training_optimizer,
        metrics=metrics_type
    )

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        shuffle=shuffle,
        validation_data=validation_data,
        batch_size=batch_size,
        verbose=0,
        callbacks=callbacks,
        # use_multiprocessing=num_workers > 1,
        # workers=num_workers
    )

    return model,history


def train_model_with_kfold(model: keras.Model,
                           X_train, y_train,
                           loss_type, training_optimizer, metrics_type,
                           epochs, shuffle, validation_data, batch_size, callbacks,
                           folder_n_splits: int = 5, folder_shuffle: bool = True):

    model_per_fold: list[keras.Model] = []
    history_per_fold: list = []
    eval_per_fold: list = []

    kfold = KFold(
        n_splits=folder_n_splits,
        shuffle=folder_shuffle,
        random_state=123
    )
    for train, test in kfold.split(X_train, y_train):
        fold_model = keras.models.clone_model(model)
        fold_model.compile(
            loss=loss_type,
            optimizer=training_optimizer,
            metrics=metrics_type
        )

        history = fold_model.fit(
            x=X_train[train], y=y_train[train],
            epochs=epochs,
            shuffle=shuffle,
            validation_data=validation_data,
            batch_size=batch_size,
            verbose=0,
            callbacks=callbacks,
            # use_multiprocessing=num_workers > 1,
            # workers=num_workers
        )

        scores = fold_model.evaluate(
            X_train[test], y_train[test], verbose=0)

        model_per_fold.append(fold_model)
        history_per_fold.append(history)
        eval_per_fold.append(scores[0])

    best_model_index = np.argmax(eval_per_fold)

    return model_per_fold[best_model_index], history_per_fold[best_model_index]


__all__=["train_model","train_model_with_kfold"]