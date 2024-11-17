import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from common.manage_log import *

def training_the_model(parameters, model, dataloader_train, dataloader_valid):

    # setting parameters for model
    # max_steps = parameters['neural_network_model']['max_steps']
    max_epochs = parameters['neural_network_model']['number_epochs']
    gradient_clip_val = parameters['neural_network_model']['gradient_clip_val']
    # logging_info(f'max_steps: {max_steps}')
    logging_info(f'max_epochs: {max_epochs}')
    logging_info(f'gradient_clip_val: {gradient_clip_val}')

    # training model
    # trainer = Trainer(max_epochs=max_epochs,
    #                   gradient_clip_val=gradient_clip_val)
    early_stopping = EarlyStopping(monitor="validation_loss", 
                                   mode="min", 
                                   verbose=True,
                                   patience=parameters['neural_network_model']['patience'],
                                   min_delta=parameters['neural_network_model']['min_delta'],
                                   )
    trainer = Trainer(max_epochs=max_epochs,
                      gradient_clip_val=gradient_clip_val, 
                      callbacks=[early_stopping],
                      )                      
    trainer.fit(model, dataloader_train, dataloader_valid)

    # saving model and state of the trained model
    path_and_model_filename = os.path.join(
        parameters['training_results']['weights_folder'], 
        parameters['training_results']['weights_base_filename']
    )
    torch.save(model.state_dict(), path_and_model_filename)

    # returning model trained
    return model