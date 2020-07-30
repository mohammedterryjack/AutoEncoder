############   NATIVE IMPORTS  ###########################
from typing import List
############ INSTALLED IMPORTS ###########################
from sklearn.neural_network import MLPRegressor
from numpy import zeros, array
############   LOCAL IMPORTS   ###########################
##########################################################
    
class AutoEncoder:
    """
    uses two FFNN from sklearn to create an Autoencoder
    """
    def __init__(
        self, 
        training_vectors:List[array],
        hidden_layer_sizes_encoder_only:List[int] = [1024,512,256,128,64,32,16,8,4],
        projection_dimension:int = 2
    ) -> None:

        self.encoder = self.get_encoder_from_autoencoder(
            vectors=training_vectors,
            trained_encoder_decoder = self.train_autoencoder(
                vectors=training_vectors,
                hidden_layer_sizes = hidden_layer_sizes_encoder_only + [projection_dimension] + hidden_layer_sizes_encoder_only[::-1],
            ),
            hidden_layer_sizes = hidden_layer_sizes_encoder_only,
            output_layer_size = projection_dimension
        )

    def reduce_dimensions(self,vectors:List[array]) -> List[array]:
        return self.encoder.predict(vectors)
    
    @staticmethod
    def train_autoencoder(vectors:List[array],hidden_layer_sizes:List[int]) -> MLPRegressor:
        model = MLPRegressor(
            random_state=1, 
            activation="relu",
            hidden_layer_sizes = hidden_layer_sizes,
            verbose=True,
            max_iter=1000,
        ) 
        model.fit(vectors,vectors)
        return model
    
    @staticmethod
    def get_encoder_from_autoencoder(
        vectors:List[array], 
        trained_encoder_decoder:MLPRegressor,
        hidden_layer_sizes:List[int],
        output_layer_size:int,
    ) -> MLPRegressor:

        model = MLPRegressor(
            random_state=1, 
            activation="relu",
            hidden_layer_sizes = hidden_layer_sizes
        ) 
        model.fit(
            X=vectors,
            y=zeros(shape=(
                len(vectors),
                output_layer_size
            ))
        )
        model.coefs_ = trained_encoder_decoder.coefs_[:len(hidden_layer_sizes)+1]
        return model