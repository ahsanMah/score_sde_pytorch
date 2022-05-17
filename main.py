# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Training and evaluation"""
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
import warnings

warnings.filterwarnings("ignore")
gpus = tf.config.list_physical_devices("GPU")
print("GPUS:", gpus)
if gpus:
    # tf.config.experimental.set_visible_devices(gpus[0], "GPU")
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        # Currently, memory growth needs to be the same across GPUs
        # for gpu in gpus:
        logical_gpus = tf.config.experimental.list_logical_devices("GPU")
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print("Could not set memory growth w/ exception:", e)

from absl import app, flags
from ml_collections.config_flags import config_flags
import logging
import run_lib

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    "config", None, "Training configuration.", lock_config=True
)
flags.DEFINE_string("workdir", None, "Work directory.")
flags.DEFINE_enum(
    "mode",
    None,
    ["train", "eval", "score", "train_dgmm"],
    "Running mode: train or eval",
)
flags.DEFINE_string(
    "eval_folder", "eval", "The folder name for storing evaluation results"
)
flags.mark_flags_as_required(["workdir", "config", "mode"])


def main(argv):
    if FLAGS.mode == "train":
        # Create the working directory
        tf.io.gfile.makedirs(FLAGS.workdir)
        # Set logger so that it outputs to both console and file
        # Make logging work for both disk and Google Cloud Storage
        gfile_stream = open(os.path.join(FLAGS.workdir, "stdout.txt"), "w")
        handler = logging.StreamHandler(gfile_stream)
        formatter = logging.Formatter(
            "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler)
        logger.setLevel("INFO")
        # Run the training pipeline
        run_lib.train(FLAGS.config, FLAGS.workdir)
    elif FLAGS.mode == "eval":
        # Run the evaluation pipeline
        run_lib.evaluate(FLAGS.config, FLAGS.workdir, FLAGS.eval_folder)
    elif FLAGS.mode == "score":
        # Run the msma scorer pipeline
        run_lib.compute_scores(FLAGS.config, FLAGS.workdir)
    elif FLAGS.mode == "train_dgmm":
        # Run the depp gmm training pipeline
        run_lib.dgmm_trainer(FLAGS.config, FLAGS.workdir)
    else:
        raise ValueError(f"Mode {FLAGS.mode} not recognized.")


if __name__ == "__main__":
    app.run(main)
