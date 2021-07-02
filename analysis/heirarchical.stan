data {
  int<lower=1> trial;
  int<lower=1, upper=2> action[trial];
  int<lower=0, upper=1> reward[trial];
}

parameters {
  real<lower=0., upper=1.> alpha;
  real<lower=0.> beta_;
  real<lower=0.> w;
}

model {
  vector[2] q_value = [0, 0]';
  vector[2] curiosity = [0, 0]';
  vector[2] td_err = [0, 0]';
  matrix[2, 2] onehot = [[1, 0], [0, 1]];
  vector[2] pi_;
  vector[2] action_;

  for (t in 1:trial) {
    action_ = onehot[action[t]]';
    pi_ = softmax(beta_ * (q_value + w * curiosity));
    // row_vector * vector => scalar, same as `sum(vector .* vector)`
    target += log(pi_' * action_);

    td_err = reward[t] - q_value;
    q_value += alpha * td_err .* action_;
    curiosity += alpha * (fabs(td_err) - curiosity) .* action_;
  }
}

generated quantities {
  vector[trial] ll;

  vector[2] q_value = [0, 0]';
  vector[2] curiosity = [0, 0]';
  vector[2] td_err = [0, 0]';
  matrix[2, 2] onehot = [[1, 0], [0, 1]];
  vector[2] pi_;
  vector[2] action_;

  for (t in 1:trial) {
    action_ = onehot[action[t]]';
    pi_ = softmax(beta_ * (q_value + w * curiosity));
    ll[t] = log(pi_' * action_);

    td_err = reward[t] - q_value;
    q_value += alpha * td_err .* action_;
    curiosity += alpha * (fabs(td_err) - curiosity) .* action_;
  }
}
