
import argparse
import json
import os

import numpy as np
from ray.air.result import Result
import tensorflow as tf

from ray.train.tensorflow import TensorflowTrainer
from ray.air.integrations.keras import Callback as TrainCheckpointReportCallback
from ray.air.config import ScalingConfig,RunConfig
from ray.air.integrations.mlflow import MLflowLoggerCallback


import tensorflow_io as tfio

from tensorflow.keras import Model
from tensorflow.keras.layers import Dropout, Dense, Input,BatchNormalization
from tensorflow.keras import Sequential
from keras.optimizers import Adam



num_features = 28
num_timesteps =11000000



def make_common_ds(batch_size: int, train_names:list) -> tf.data.Dataset:
    
   
    columns_init = {"leptonpT": tf.TensorSpec(tf.TensorShape([]), tf.double), 
                "leptoneta": tf.TensorSpec(tf.TensorShape([]), tf.double),
                "leptonphi": tf.TensorSpec(tf.TensorShape([]), tf.double),
                "magnitudfaltante": tf.TensorSpec(tf.TensorShape([]), tf.double), 
                "faltaenergiaphi": tf.TensorSpec(tf.TensorShape([]), tf.double),
                "jet1pt": tf.TensorSpec(tf.TensorShape([]), tf.double),
                "jet1eta": tf.TensorSpec(tf.TensorShape([]), tf.double), 
                "jet1phi": tf.TensorSpec(tf.TensorShape([]), tf.double),
                "jet1btag": tf.TensorSpec(tf.TensorShape([]), tf.double),
                "jet2pt": tf.TensorSpec(tf.TensorShape([]), tf.double), 
                "jet2eta": tf.TensorSpec(tf.TensorShape([]), tf.double),
                "jet2phi": tf.TensorSpec(tf.TensorShape([]), tf.double),
                "jet2btag": tf.TensorSpec(tf.TensorShape([]), tf.double), 
                "jet3pt": tf.TensorSpec(tf.TensorShape([]), tf.double),
                "jet3eta": tf.TensorSpec(tf.TensorShape([]), tf.double),
                "jet3phi": tf.TensorSpec(tf.TensorShape([]), tf.double), 
                "jet3btag": tf.TensorSpec(tf.TensorShape([]), tf.double),
                "jet4pt": tf.TensorSpec(tf.TensorShape([]), tf.double),
                "jet4eta": tf.TensorSpec(tf.TensorShape([]), tf.double), 
                "jet4phi": tf.TensorSpec(tf.TensorShape([]), tf.double),
                "jet4btag": tf.TensorSpec(tf.TensorShape([]), tf.double),
                "m_jj": tf.TensorSpec(tf.TensorShape([]), tf.double), 
                "m_jjj": tf.TensorSpec(tf.TensorShape([]), tf.double),
                "m_lv": tf.TensorSpec(tf.TensorShape([]), tf.double),
                "m_jlv": tf.TensorSpec(tf.TensorShape([]), tf.double), 
                "m_bb": tf.TensorSpec(tf.TensorShape([]), tf.double),
                "m_wbb": tf.TensorSpec(tf.TensorShape([]), tf.double),
                "m_wwbb": tf.TensorSpec(tf.TensorShape([]), tf.double),
                "y": tf.TensorSpec(tf.TensorShape([]), tf.double)}

   
   
    ds= (
        train_names.interleave(
            lambda f: tfio.IODataset.from_parquet(f, columns=columns_init),
            cycle_length=28,
            block_length=28,
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    ).batch(batch_size)

    # get all columns from dataset tensorflow except y
    features = ds.map(lambda x: {k: v for k, v in x.items() if k != 'y'})

    # get column from dataset tensorflow
    labels = ds.map(lambda x: x['y'])
    
       
    train_dataset=(        
            tf.data.Dataset.zip((features, labels)).interleave(
            lambda f, l: tf.data.Dataset.from_tensor_slices((f, l)),
            cycle_length=28,
            block_length=28,
            num_parallel_calls=tf.data.AUTOTUNE, 
            
        ).batch(batch_size)                
        .shuffle(buffer_size = batch_size)
        .repeat()
        .batch(batch_size)
        .prefetch(batch_size)
        .map(pack_features_vector)
        
    )
  
    return train_dataset


def pack_features_vector(features, labels):
    """Pack the features into a single array."""    

    features = tf.stack(list(features.values()), axis=2)
    return features, labels




def build_Seq_model(config: dict) -> tf.keras.Model:

    ip = Input(shape=(None,num_features))  
    x= BatchNormalization()(ip)  
    x = Sequential(Dense(config.get('u1'), activation='relu'))(ip)
    x = Dropout(0.8)(x)
    x = Sequential(Dense(config.get('u2'), activation='relu'))(x)
    x = Dropout(0.8)(x)
    x = Sequential(Dense(config.get('u3'), activation='relu'))(x)
    x = Dropout(0.8)(x)
    out = Dense(1, activation='softmax')(x)

    model = Model(ip, out)
    
    return model

def train_func(config: dict):
    per_worker_batch_size = config.get("batch_size", 28)
    epochs = config.get("epochs", 3)
    steps_per_epoch = config.get("steps_per_epoch", 170)

    tf_config = json.loads(os.environ["TF_CONFIG"])
    num_workers = len(tf_config["cluster"]["worker"])
    #https://www.tensorflow.org/api_docs/python/tf/distribute/MultiWorkerMirroredStrategy
    communication_options = tf.distribute.experimental.CommunicationOptions(
    implementation=tf.distribute.experimental.CommunicationImplementation.AUTO)
    
    strategy = tf.distribute.MultiWorkerMirroredStrategy()

    global_batch_size = per_worker_batch_size * num_workers
    

    train_names = tf.data.Dataset.list_files(f'/home/hvargas/Documentos/stratio/proyectos/sas/data/csv_to_parquet/data/*.parquet')

    multi_worker_dataset = make_common_ds(global_batch_size,train_names)

    with strategy.scope():
        # Model building/compiling need to be within `strategy.scope()`.
        multi_worker_model = build_Seq_model(config)       
        
        multi_worker_model.compile(
        optimizer='adam',
            loss='binary_crossentropy',
            metrics=["accuracy"],
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='loss',
            verbose=1,
            patience=300,
            mode='min',
            restore_best_weights=True)
    
    history = multi_worker_model.fit(
        multi_worker_dataset,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        callbacks=[TrainCheckpointReportCallback(),early_stopping],
    )
    results = history.history
    return results


def train_tensorflow_sas(
    num_workers: int = 1, use_gpu: bool = False, epochs: int = 4,
    u1: int = 8,u2: int = 8,dout2: float = 0.8,u3: int = 8,dout3: float = 0.8,
    learning_rate: float = 0.001,patience: int = 10
) -> Result:
    config = {"lr": 1e-3, "batch_size": 28, "epochs": epochs,"u1": u1,"u2": u2,
                "dout2": dout2,"u3": u3,"dout3": dout3,"learning_rate": learning_rate,
                "patience": patience}
    trainer = TensorflowTrainer(
        train_loop_per_worker=train_func,
        train_loop_config=config,
        scaling_config=ScalingConfig(num_workers=num_workers, use_gpu=use_gpu),
        run_config=RunConfig(
            callbacks=[MLflowLoggerCallback(experiment_name="train_sas")]
        ),
    )
    results = trainer.fit()
    print("Final metrics: ", results.metrics)
    return results
  
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--address", required=False, type=str, help="the address to use for Ray"
    )
    parser.add_argument(
        "--num-workers",
        "-n",
        type=int,
        default=1,
        help="Sets number of workers for training.",
    )
    parser.add_argument(
        "--use-gpu", action="store_true", default=False, help="Enables GPU training"
    )
    parser.add_argument(
        "--epochs", type=int, default=3, help="Number of epochs to train for."
    )
    parser.add_argument(
        "--u1", type=int, default=2048, help="Number of units in the first layer."
    )
    parser.add_argument(
        "--u2", type=int, default=2048, help="Number of units in the second layer."
    )
    parser.add_argument(
        "--dout2", type=int, default=0.4, help="Dropout rate in the second layer."
    )
    parser.add_argument(
        "--u3", type=int, default=2048, help="Number of units in the third layer."
    )
    parser.add_argument(
        "--dout3", type=int, default=0.3, help="Dropout rate in the third layer."
    )    

    parser.add_argument(
        "--learning_rate", type=float, default= 0.01, help="Learning rate for training."
    )
    
    parser.add_argument(
        "--patience", type=int, default=10, help=" Patience"
    )
    

    args, _ = parser.parse_known_args()

    import ray

    ray.init(address=args.address)
    train_tensorflow_sas(
        num_workers=args.num_workers, 
        use_gpu=args.use_gpu, 
        epochs=args.epochs,
        u1=args.u1,
        u2=args.u2,
        dout2=args.dout2,
        u3=args.u3,
        dout3=args.dout3,
        learning_rate=args.learning_rate,
        patience=args.patience

    )

