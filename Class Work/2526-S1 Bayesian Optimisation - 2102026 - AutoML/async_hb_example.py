import time,os,json,sys
from pathlib import Path

import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import hinge_loss

import ray
from ray import tune
from ray.tune import loguniform
from ray.tune import Trainable
from ray.tune.search.optuna import OptunaSearch # Import OptunaSearch
from optuna.samplers import RandomSampler # Import RandomSampler for Optuna
#from ray.tune.schedulers import HyperBandScheduler # Import HyperBand Scheduler
from ray.tune.schedulers import ASHAScheduler # Import Asynchronous Hyperband Scheduler


### CONSTANTS ###
RANDOM_SEED = 12345
TEST_SPLIT = 0.2
METRIC = "loss"


# Generate data for classification
# Here we create a synthetic binary classification dataset
# We will use this dataset to train a basic linear SVM model
# and optimize its hyperparameters using an asynchronous Hyperband with random search strategy.
X, y = make_classification(
    n_samples=1000, 
    n_features=5, 
    n_informative=3,
    n_classes=2,
    n_redundant=2, random_state=RANDOM_SEED
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SPLIT, random_state=RANDOM_SEED
)


class BasicSVM(Trainable):
    r"""
    This is a basic linear SVM Trainable class for Ray Tune BOHB example.
    """

    def setup(self, config):
        r"""
        Setup function for the Trainable class.

        Args:
            config (dict): Configuration dictionary containing hyperparameters.
        
        Returns:
            None
        """
        self.alpha = config.get("alpha", 0.0001)
        self.max_iter = config.get("max_iter", 1000)
        self.eta_0 = config.get("eta_0", 0.01)
        self.batch_size = 64

        # Random generator for batching
        self.rng = np.random.default_rng(RANDOM_SEED)

        self.clf = SGDClassifier(
            loss="hinge",
            alpha=self.alpha,
            learning_rate="optimal",
            random_state=RANDOM_SEED,
            eta0=self.eta_0,
        )

        # Initial call requires providing class labels
        self.classes = np.unique(y_train)
        self.timestep = 0
        self.steps_performed = 0

    def step(self):
        r"""
        Single training step for the Trainable class. This is similar to the training step of 
        a Neural Network in any framework (e.g. PyTorch, TensorFlow, etc.).
        Here, we perform one epoch of training on a batch of data and evaluate the model
        on the validation set.

        Returns:
            dict: Dictionary containing the accuracy and f1_score on the validation set.
        """

        self.timestep += 1
        
        # Train for 1 batch per step
        for _ in range(len(X_train) // self.batch_size):
            idx = self.rng.choice(len(X_train), self.batch_size)
            X_batch = X_train[idx]
            y_batch = y_train[idx]

            # Use partial_fit for online learning
            self.clf.partial_fit(X_batch, y_batch, classes=self.classes)
            self.steps_performed += 1

        # Evaluate on a small validation split for speed
        loss = hinge_loss(y_train, self.clf.predict(X_train))
        accuracy = accuracy_score(y_test, self.clf.predict(X_test))
        f1_val = f1_score(y_test, self.clf.predict(X_test), labels=self.classes,
                           average='binary')

        return {"loss":loss,"accuracy": accuracy, "f1_score":f1_val, "steps": self.steps_performed, "training_iteration": self.timestep}

    def save_checkpoint(self, checkpoint_dir):
        r"""
        Save checkpoint for the Trainable class.
        Args:
            checkpoint_dir (str): Directory to save the checkpoint.
        """
        path = os.path.join(checkpoint_dir, "checkpoint")
        with open(path, "w") as f:
            f.write(json.dumps({"timestep": self.timestep}))

    def load_checkpoint(self, checkpoint_dir):
        path = os.path.join(checkpoint_dir, "checkpoint")
        with open(path, "r") as f:
            self.timestep = json.loads(f.read())["timestep"]



# Main execution
if __name__ == "__main__":

    # Initialize Ray
    ray.init(num_cpus=2)

    # Initial configuration
    config = {
        "alpha": loguniform(1e-5, 1e-1),
        "eta_0": loguniform(1e-3, 1e-1),
        "max_steps": 100,
    }

    
    max_iterations = 10

    # Import an Optuna Random Search algorithm
    # This is the major modification compared to BOHB example
    # Since the samples are purely random instead of using Bayesian Optimization
    # to generate new samples based on previous results
    random_search_optuna = OptunaSearch(
                                        sampler=RandomSampler(seed=RANDOM_SEED))

    # Limit the number of concurrent trials (This is to avoid overloading the system)
    random_search_optuna = tune.search.ConcurrencyLimiter(random_search_optuna, max_concurrent=4)

    # Import an Hyperband Scheduler
    scheduler = ASHAScheduler(
        time_attr="training_iteration",
        max_t=100,
        reduction_factor=2,
        stop_last_trials=False,
    )

    tuner = tune.Tuner(
        BasicSVM,
        run_config=tune.RunConfig(
            name="hyperband_test", 
            stop={"training_iteration": 100},
            storage_path=str(Path.cwd() / "ray_results"),
        ),
        tune_config=tune.TuneConfig(
            metric=METRIC,
            mode="max",
            scheduler=scheduler,
            search_alg=random_search_optuna,
            num_samples=16,
        ),
        param_space=config,
    )

    results = tuner.fit()

    print("Best hyperparameters found were: ", results.get_best_result().config,
        "with loss=", results.get_best_result().metrics["loss"],
        "with accuracy=", results.get_best_result().metrics["accuracy"],
        "and f1_score=", results.get_best_result().metrics["f1_score"])
    

