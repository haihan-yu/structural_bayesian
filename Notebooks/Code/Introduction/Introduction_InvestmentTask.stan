// Introduction_InvestmentTask.stan
data {
    int<lower=0> n; // number of observations
    int<lower=0> ny; // size of the choice set
    int y[n];   // choices made by participant
    vector[n] X;   // multiplier for investment
    vector[ny] ygrid; // possible values of y (i.e. the choice set)
    real prior_r[2]; // prior mean and sd for r
    real prior_lambda[2]; // prior mean and sd for log(lambda)
}

parameters {
    real r; // parameter in CRRA utility function x^(1-r)/(1-r)
    real<lower=0> lambda; // logit choice precision
}

model {
    for (ii in 1:n) {
        vector[ny] EU; // expected utility of choosing each action
        vector[ny] lpr; // (log) probability of choosing each action

        for (j in 1:ny) {
            EU[j] = 0.5 * pow((100 - ygrid[j]), (1.0-r)) / (1.0-r) + 0.5 * pow((100 + ygrid[j] * (X[ii] - 1)), (1.0-r)) / (1.0-r);
        }

        lpr = log_softmax(lambda*EU);
        target += lpr[y[ii]];
    }

    // specify the priors for the parameters
    r ~ normal(prior_r[1],prior_r[2]);
    lambda ~ lognormal(prior_lambda[1],prior_lambda[2]);
}