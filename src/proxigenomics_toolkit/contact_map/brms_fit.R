suppressMessages(library(readr))
suppressMessages(library(dplyr))
suppressMessages(library(sjmisc))
suppressMessages(library(tibble))
# suppressMessages(library(car))
suppressMessages(library(PerformanceAnalytics))
suppressMessages(library(DHARMa))
# suppressMessages(library(stats))
suppressMessages(library(brms))

dharma_data <- function(fit, newdata=NULL, samples=1000, seed=12345) {
    set.seed(seed)
    if (is.null(newdata)) {
        x <- sample_n(fit$data, samples)
    }
    else {
        x <- sample_n(newdata, samples)
    }
    createDHARMa(
        simulatedResponse = t(posterior_predict(fit, newdata=x, cores=8)),
        observedResponse = x$contacts1m,
        fittedPredictedResponse = apply(t(posterior_epred(fit, newdata=x, cores=8)), 1, mean),
        integerResponse = T
    )
}

folded_power <- function(x,p) { x^p - (1-x)^p}

calc_lambda <- function(var_) {
    bc <- MASS::boxcox(var_ ~ 1, plotit = F, interp=T, seq(-5, 20, 1/50))
    bc$x[which.max(bc$y)]
}

boxcox_transform <- function(var_, lam_) {
    if (lam_ == 0) {
        log(var_)
    }
    else{
        (var_^lam_ - 1) / lam_
    }
}

get_sample <- function(data, n_samples, seed) {
    if (n_samples < nrow(data)) {
        writeLines(paste('Reducing observation set size'))
        sample_n(data, n_samples, seed = seed)
    }
    else {
        writeLines(paste('Using all observations'))
        data
    }
}

handle_nas <- function(df, remove_na) {
    num_na = sum(is.na(df))
    if (num_na > 0) {
        writeLines(paste('Transformed data contained', num_na, 'NAs'))
        if (remove_na) {
            writeLines('Removing rows containing NAs')
            df <- na.omit(df)
        }
        else {
            writeLines('Setting NAs to zero')
            df[is.na(df)] <- 0
        }
    }
    df
}

transform_data <- function(data, robust_='sd', power_=1/3, type="log", lambdas=NULL) {

    # drop rows with zeros
    d <- tibble(subset(data,
                       sites_u > 0 & sites_v > 0 &
                       cov_u > 0 & cov_v > 0 &
                       uf_u > 0 & uf_v > 0))

    # make sure the columns are full precision to minimise overflows
    d <- dplyr::mutate(d,
                       across(
                           c(length_u,
                             length_v,
                             sites_u,
                             sites_v,
                             cov_u,
                             cov_v,
                             uf_u,
                             uf_v),
                           as.double))

    d <- dplyr::mutate(d,
                       # response (endogenous) variable
                       # ZIF model by 1s -> 0s
                       contacts1m = contacts - 1,
                       # boxcox transform exogenous variables
                       length = length_u * length_v,
                       coverage = cov_u * cov_v,
                       sites = sites_u * sites_v,
                       density = sites_u/length_u * sites_v/length_v,
                       # proportional exogenous variables
                       uniqueness = uf_u * uf_v)

   if (type == "boxcox") {
       if (is.null(lambdas)) {
           lambdas <- list(length = calc_lambda(d$length),
                           coverage = calc_lambda(d$coverage),
                           sites = calc_lambda(d$sites),
                           density = calc_lambda(d$density),
                           uniqueness = calc_lambda(d$uniqueness))
       }

       d <- dplyr::mutate(d,
                     # response (endogenous) variable
                     # ZIF model by 1s -> 0s
                     contacts1m = contacts - 1,
                     # boxcox transform exogenous variables
                     length = boxcox_transform(length, lambdas$length),
                     coverage = boxcox_transform(coverage, lambdas$coverage),
                     sites = boxcox_transform(sites, lambdas$sites),
                     density = boxcox_transform(density, lambdas$density),
                     # proportional exogenous variables
                     uniqueness = folded_power(uniqueness, power_)
                     # uniqueness = boxcox_transform(uniqueness, lambdas$uniqueness),
                     # uniqueness = asin(sqrt(uniqueness))
       )
   }
   else if (type == "log") {
       d <- dplyr::mutate(d,
                     # response (endogenous) variable
                     # ZIF model by 1s -> 0s
                     contacts1m = contacts - 1,
                     # log transform
                     length = log(length),
                     coverage = log(coverage),
                     sites = log(sites),
                     density = log(density),
                     # proportional exogenous variables
                     uniqueness = folded_power(uniqueness, power_)
                     # uniqueness = boxcox_transform(uniqueness, lambdas$uniqueness),
                     # uniqueness = asin(sqrt(uniqueness))
       )
   }

    # Prepare data, where sequence (u) and genome_bin (v) variables are taken as products
    d <- dplyr::select(
        sjmisc::std(d,
            # standardise exogenous
            length,
            coverage,
            sites,
            density,
            uniqueness,
            robust = robust_),
        # drop unstandardised exog columns
        -c('length', 'coverage', 'sites', 'density', 'uniqueness'))

    # return
    if (type == "boxcox") {
        list(data = d, lambdas = lambdas)
    }
    else if (type == "log") {
        list(data = d)
    }
}

brms_fit2 <- function(spurious, all_contacts, output_path, n_samples, seed,
                     fixed_model, shape_model, n_iterations=2500, validate=TRUE, remove_na=TRUE) {

    MAX_POINTS <- 1000

    if (missing(output_path)) {
        output_path <- "."
    }
    if (!missing(seed)) {
        set.seed(seed)
    }

    spurious <- rowid_to_column(spurious, 'ID') %>%
        select(ID, seq, cluster, cluster_name, size_v, contacts,
               length_u, length_v, cov_u, cov_v, sites_u, sites_v, gc_u, gc_v, uf_u, uf_v, intra)

    writeLines(paste('Input rows:', nrow(spurious)))

    spurious <- transform_data(spurious)
    spurious$data <- handle_nas(spurious$data, remove_na)
    writeLines(paste('Tranformed rows:', nrow(spurious$data)))

    spurious$data <- dplyr::filter(spurious$data, !intra)
    writeLines(paste('Observations available for sampling:', nrow(spurious$data)))

    if (missing(n_samples)) {
        n_samples <- nrow(spurious$data)
    }

    dfit <- get_sample(spurious$data, n_samples, seed)
    writeLines(paste('Fitting with:', nrow(dfit)))

    # plot points
    if (nrow(dfit) < MAX_POINTS) {
        n_points <- nrow(dfit)
    } else {
        n_points <- MAX_POINTS
    }

    writeLines('Creating correlation plot')
    png(paste0(output_path, '_R_correlation.png'), width = 1200, height = 800)
    chart.Correlation(sample_n(dfit[, c('contacts1m', 'sites_z', 'length_z',
                                        'coverage_z', 'density_z', 'uniqueness_z')], n_points),
                      histogram = TRUE, pch = 19)
    dev.off()

    writeLines('Fitting model')
    model_formula <- bf(as.formula(fixed_model), as.formula(shape_model))
    #contacts1m ~ length_z*coverage_z + length_z*density_z + length_z*uniqueness_z + I(length_z^2)
    #shape ~ length_z + coverage_z + density_z + uniqueness_z

    model_priors <- c(prior(student_t(3, 0, 2.5), class=b),
                      prior(student_t(3, 0, 3), dpar=shape),
                      # prior(student_t(3, 0, 10), dpar=shape, class=Intercept),
                      prior(beta(1, 2), class=zi))

    model <- brm(model_formula, prior = model_priors, family = zero_inflated_negbinomial(),
                 cores = 4, iter = n_iterations, warmup = 1000,
                 data = spurious$data, seed=seed, backend='cmdstanr',)

    fit_summary <- summary(model)
    writeLines('\nSummary for model fit')
    print(fit_summary)
    # writeLines('\nParameter confidence intervals')
    # print(confint(model))
    sink(paste0(output_path, '_R_brms_fit.log'), append = TRUE)
    sink()

    if (validate) {
        # simulate residuals plot for model quality inspection
        writeLines('\nCreating DHARMA data object')
        pdf(paste0(output_path, '_R_brms_validation.pdf'))
        layout.matrix <- matrix(c(1, 3, 2, 4), nrow = 2, ncol = 2)
        layout(mat = layout.matrix, heights = c(1, 1), widths = c(1, 1))
        par(cex.lab=0.67)
        simOut <- dharma_data(model, samples=n_points, seed=seed)
        writeLines('\nTest uniformity')
        print(testUniformity(simOut, plot=F))
        plotQQunif(simOut, testUniformity = T, testOutliers = F, testDispersion = F)
        writeLines('\nTest quantiles')
        print(testQuantiles(simOut))
        writeLines('\nTest dispersion')
        print(testDispersion(simOut))
        writeLines('\nTest zero-inflation')
        print(testZeroInflation(simOut))
        writeLines('\nTest outliers with boostrapping')
        print(testOutliers(simOut, type = 'bootstrap'))
        dev.off()
    }

    writeLines('Calculating predictions for spurious contacts')
    spurious$data <- bind_cols(spurious$data,
                            as_tibble(predict(model,
                                              cores=4,
                                              probs=c(0.68, 0.95, 0.98),
                                              robust=T)))

    writeLines('Calculating predictions for all contacts')
    all_contacts <- rowid_to_column(all_contacts, 'ID') %>%
        select(ID, seq, cluster, cluster_name, size_v, contacts,
               length_u, length_v, cov_u, cov_v, sites_u, sites_v, gc_u, gc_v, uf_u, uf_v, intra)

    all_contacts <- transform_data(all_contacts, lambdas=spurious$lambdas)
    # all_contacts$data <- bind_cols(all_contacts$data,
    #                                as_tibble(predict(model,
    #                                                  newdata=all_contacts$data,
    #                                                  cores=4,
    #                                                  probs=c(0.68, 0.95, 0.98),
    #                                                  robust=T)))

    list(fitted = spurious$data, all_contacts = all_contacts$data,
         model = model, lambdas = spurious$lambdas, fit_type='brms')
}

