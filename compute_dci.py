# coding=utf-8
# Copyright 2018 The DisentanglementLib Authors.  All rights reserved.
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

"""Implementation of Disentanglement, Completeness and Informativeness.

Based on "A Framework for the Quantitative Evaluation of Disentangled
Representations" (https://openreview.net/forum?id=By-7dz-AZ).
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl import logging
import numpy as np
import scipy
from six.moves import range
from sklearn import ensemble


def compute_dci(mus_train, ys_train):
  """Computes score based on both training and testing codes and factors."""
  scores = {}
  importance_matrix, train_err = compute_importance_gbt(
      mus_train, ys_train)
  assert importance_matrix.shape[0] == mus_train.shape[0]
  assert importance_matrix.shape[1] == ys_train.shape[0]
  scores["informativeness_train"] = train_err
  scores["disentanglement"] = disentanglement(importance_matrix)
  scores["completeness"] = completeness(importance_matrix)
  return scores

# x_train: [num_codes, num_train]
# y_train: [num_factors, num_train]

def compute_importance_gbt(x_train, y_train):
  """Compute importance based on gradient boosted trees."""
  num_factors = y_train.shape[0]
  num_codes = x_train.shape[0]
  importance_matrix = np.zeros(shape=[num_codes, num_factors],
                               dtype=np.float64)
  train_loss = []
  test_loss = []
  for i in range(num_factors):
    model = ensemble.GradientBoostingClassifier()
    model.fit(x_train.T, y_train[i, :])
    importance_matrix[:, i] = np.abs(model.feature_importances_)
    train_loss.append(np.mean(model.predict(x_train.T) == y_train[i, :]))
  return importance_matrix, np.mean(train_loss)


def disentanglement_per_code(importance_matrix):
  """Compute disentanglement score of each code."""
  # importance_matrix is of shape [num_codes, num_factors].
  return 1. - scipy.stats.entropy(importance_matrix.T + 1e-11,
                                  base=importance_matrix.shape[1])


def disentanglement(importance_matrix):
  """Compute the disentanglement score of the representation."""
  per_code = disentanglement_per_code(importance_matrix)
  if importance_matrix.sum() == 0.:
    importance_matrix = np.ones_like(importance_matrix)
  code_importance = importance_matrix.sum(axis=1) / importance_matrix.sum()

  return np.sum(per_code*code_importance)


def completeness_per_factor(importance_matrix):
  """Compute completeness of each factor."""
  # importance_matrix is of shape [num_codes, num_factors].
  return 1. - scipy.stats.entropy(importance_matrix + 1e-11,
                                  base=importance_matrix.shape[0])


def completeness(importance_matrix):
  """"Compute completeness of the representation."""
  per_factor = completeness_per_factor(importance_matrix)
  if importance_matrix.sum() == 0.:
    importance_matrix = np.ones_like(importance_matrix)
  factor_importance = importance_matrix.sum(axis=0) / importance_matrix.sum()
  return np.sum(per_factor*factor_importance)

if __name__ == "__main__":
    import numpy as np
    import glob
    fn = "igibson_mask_s3/traj*.npz"

    for filename in glob.glob("test_disentanglement/" + fn):
        ts = np.load(filename)

    skill = ts.get("skill")
    obs = ts.get("obs")
    obs = obs[:, 0, :]
    n_points = obs.shape[0]
    print(f"evaluating on {n_points} data")

    code = obs

    x = code.T
    y = skill.T

    print(compute_dci(x, y))

