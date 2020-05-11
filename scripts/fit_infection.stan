functions {
    /* discretesized version of lognormal distribution */
    vector plognormal(real mu, real sigma, int K) {
        vector[K] res; 
        for (k in 1:K)
            res[k] = exp(lognormal_lcdf(k | mu, sigma));

        return append_row(res[1], res[2:K]-res[1:(K-1)]);        
    }

    /* discretesized version of Weibull distribution */
    vector pweibull(real kappa, real theta, int K) {
        vector[K] res; 
        for (k in 1:K)
            res[k] = -expm1(-pow(1.0*k/theta, kappa));

        return append_row(res[1], res[2:K]-res[1:(K-1)]);  
    }

    /* calculating the convolutions */
    // X: first function, Yrev: reversed version of the second function
    // K: length of X and Yrev
    // the result is the vector of length K-1, because the first element equal zero is omitted
    vector convolution(vector X, vector Yrev, int K) {
        vector[K-1] res;
        res[1] = X[1]*Yrev[K];
        for (k in 2:K-1) 
            res[k] = dot_product(head(X, k), tail(Yrev, k));

        return res;        
    }

    // Special form of the convolution with adjusting to the delay
    // F: cumulative distribution function of the reporting delay
    vector convolution_with_delay(vector X, vector Yrev, vector F, int K) {
        vector[K-1] res;
        vector[K] Z = X ./ F;

        res[1] = F[2]*Z[1]*Yrev[K];
        for (k in 3:K) 
            res[k-1] = F[k]*dot_product(head(Z, k-1), tail(Yrev, k-1));

        return res;        
    }
}

data {
    int<lower = 1> K; //number of days
    vector<lower = 0>[K] imported_backproj;
    vector<lower = 0>[K] domestic_backproj;
    int<lower = K> upper_bound;

    // serial interval
    real<lower = 0> param1_SI;
    real<lower = 0> param2_SI;

    // reporting delay is given by Weibul distribution
    real<lower = 0> param1_delay;
    real<lower = 0> param2_delay;

    // incubation period
    real mu_inc;
    real<lower = 0> sigma_inc;
}

transformed data {
    vector[K] cases_backproj;

    vector[K-1] conv;
    vector[K-1] conv_delay_adj;

    cases_backproj = imported_backproj + domestic_backproj;
    {
        // serial interval
        vector[K] gt = pweibull(param1_SI, param2_SI, K);
        vector[K] gtrev;
 
        vector[upper_bound] IncPeriod = plognormal(mu_inc, sigma_inc, upper_bound); 
        vector[upper_bound] IncPeriod_inv;
        vector[upper_bound] ReportDelay;
        vector[upper_bound] conv_report_inc;
        
        vector[upper_bound] F_tmp;
        vector[K] F; 
        
        for (k in 1:K) 
            gtrev[k] = gt[K+1-k];
        
        for (k in 1:upper_bound)
            IncPeriod_inv[k] = IncPeriod[upper_bound+1-k];

        // vector[upper_bound] ReportDelay;
        ReportDelay = pweibull(param1_delay, param2_delay, upper_bound);

        // convolution of the reporting delay distribution and incubation period
        // vector[upper_bound] conv_report_inc;
        conv_report_inc = append_row(rep_vector(0,1), convolution(ReportDelay, IncPeriod_inv, upper_bound));
        conv_report_inc = cumulative_sum(conv_report_inc);
        
        ReportDelay = cumulative_sum(ReportDelay);
        for (k in 1:upper_bound) 
            F_tmp[k] = ReportDelay[upper_bound+1-k];
        F = head(F_tmp, K);

        // convolution without adjusting for the delay
        conv = convolution(cases_backproj, gtrev, K);

        // convolution with adjusting for the delay
        conv_delay_adj = convolution_with_delay(cases_backproj, gtrev, F, K);
    }
}

parameters {
    // effective reproduction number without adjustment to the delay in reporting
    vector<lower = 0>[K-1] Rt;

    // effective reproduction number with adjustment to the delay in reporting
    vector<lower = 0>[K-1] Rt_adj;
}

model {
    Rt ~ normal(2.4, 2.0);
    Rt_adj ~ normal(2.4, 2.0);
    
    target += gamma_lpdf(tail(domestic_backproj, K-1) | Rt .* conv + 1e-13, 1.0)
            + gamma_lpdf(tail(domestic_backproj, K-1) | Rt_adj .* conv_delay_adj + 1e-13, 1.0);
}
