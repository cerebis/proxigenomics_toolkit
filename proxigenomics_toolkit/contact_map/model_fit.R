# suppress console warnings about function name clashes
suppressMessages(library(readr))
suppressMessages(library(dplyr))
suppressMessages(library(sjmisc))
suppressMessages(library(tibble))
suppressMessages(library(glmmTMB))
suppressMessages(library(car))
suppressMessages(library(PerformanceAnalytics))
suppressMessages(library(DHARMa))
suppressMessages(library(stats))

logit_transform <- function(x) { log(x) - log(1-x) }

transform_data <- function(data, robust='sd') {

    # drop rows with zeros
    d <- tibble(subset(data, sites_u > 0 & sites_v > 0 &
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
                      gc_u,
                      gc_v,
                      uf_u,
                      uf_v),
                    as.double))

    # Prepare data, where sequence (u) and genome_bin (v) variables are taken as products
    d <- dplyr::select(
         sjmisc::std(
             dplyr::mutate(d,
                    # endogenous variable will use a ZIF model rather than one-flated
                    contacts1m = contacts - 1,
                    # log transformation of exogenous
                    length = log(length_u * length_v),
                    sites = log(sites_u * sites_v),
                    cov = log(cov_u * cov_v),
                    # arcsine transformation for proportional variables
                    uf = logit_transform(uf_u * uf_v),
                    gc = gc_v - gc_u),
             # standardise exogenous
             length,
             sites,
             cov,
             gc,
             uf,
             robust = robust),
         # drop unstandardised exog columns
         -c('length', 'sites', 'cov', 'gc', 'uf'))

    # return
    d
}

sigma_func <- function(data, fitobj) {
    disp_X <- model.matrix(fitobj$modelInfo$allForm$dispformula, data)
    disp_beta <- fixef(fitobj)$disp
    family(fitobj)$linkinv(disp_X %*% disp_beta)
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

find_significant <- function(spurious, all_contacts, output_path, distrib_func, n_samples, seed,
                             fixed_model1, disp_model1, zi_model1,
                             fixed_model2, disp_model2, zi_model2,
                             validate=TRUE, twopass=False) {
    if (twopass) {
        find_significant_twopass(spurious, all_contacts, output_path, distrib_func, n_samples, seed,
                                 fixed_model1, disp_model1, zi_model1,
                                 fixed_model2, disp_model2, zi_model2,
                                 validate=validate)
    }
    else {
        find_significant_onepass(spurious, all_contacts, output_path, distrib_func, n_samples, seed,
                                 fixed_model1, disp_model1, zi_model1,
                                 validate=validate)
    }
}

find_significant_twopass <- function(spurious, all_contacts, output_path, distrib_func, n_samples, seed,
                                     fixed_model1, disp_model1, zi_model1,
                                     fixed_model2, disp_model2, zi_model2,
                                     validate=TRUE, remove_na=TRUE) {

    MAX_POINTS <- 1000

    if (missing(distrib_func)) {
        distrib_func <- nbinom2
    }
    if (missing(output_path)) {
        output_path <- "."
    }
    if (!missing(seed)) {
        set.seed(seed)
    }

    # convert strings to formulae
    fixed_model1 <- as.formula(fixed_model1)
    disp_model1 <- as.formula(disp_model1)
    zi_model1 <- as.formula(zi_model1)
    fixed_model2 <- as.formula(fixed_model2)
    disp_model2 <- as.formula(disp_model2)
    zi_model2 <- as.formula(zi_model2)

    writeLines(paste('Input rows:', nrow(spurious)))

    spurious <- transform_data(spurious)
    spurious <- handle_nas(spurious, remove_na)
    writeLines(paste('Tranformed rows:', nrow(spurious)))

    spurious <- filter(spurious, !intra)
    writeLines(paste('Observations available for sampling:', nrow(spurious)))

    if (missing(n_samples)) {
        n_samples <- nrow(spurious)
    }

    dfit <- get_sample(spurious, n_samples, seed)
    writeLines(paste('Fitting with:', nrow(dfit)))

    # plot points
    if (nrow(dfit) < MAX_POINTS) {
        n_points <- nrow(dfit)
    } else {
        n_points <- MAX_POINTS
    }

    writeLines('Creating correlation plot')
    png(paste0(output_path, '_R_correlation.png'), width = 1200, height = 800)
    chart.Correlation(sample_n(dfit[, c('contacts1m', 'sites_z', 'length_z', 'cov_z', 'gc_z', 'uf_z')], n_points),
                      histogram = TRUE, pch = 19)
    dev.off()

    #
    # ROUND 1
    #
    # Note: prepare an initial model of our assumed spurious contacts
    #
    writeLines('Round 1 model')
    model1 <- glmmTMB(fixed_model1,
                      ziformula = zi_model1,
                      dispformula = disp_model1,
                      family = distrib_func,
                      control = glmmTMBControl(parallel = 8),
                      data = dfit, verbose = F)

    # some quality control
    fit_summary1 <- summary(model1)
    writeLines('\nSummary for model fit')
    print(fit_summary1)
    writeLines('\nParameter confidence intervals')
    print(confint(model1))
    sink(paste0(output_path, '_R_round1_fit.log'), append = TRUE)
    writeLines('\nSummary for round 1 model fit')
    print(fit_summary1)
    sink()

    if (validate) {
        # simulate residuals plot for model quality inspection
        writeLines('\nSimulating residuals')
        pdf(paste0(output_path, '_R_round1_validation.pdf'))
        layout.matrix <- matrix(c(1, 3, 2, 4), nrow = 2, ncol = 2)
        layout(mat = layout.matrix, heights = c(1, 1), widths = c(1, 1))
        par(cex.lab=0.67)
        simOut <- simulateResiduals(model1, plot = F, seed = seed, integerResponse = T, n = 1000)
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

    # predict expeced contacts (response) from model
    spurious['response'] <- predict(model1, spurious, type = 'response')
    # Define p-value as P(X>=expected), where "expected" is that predicted from model
    spurious['pvalue'] <- 1 - pnbinom(spurious$contacts1m,
                                      size = sigma_func(spurious, model1),
                                      mu = spurious$response)
    spurious['adj_pvalue'] <- p.adjust(spurious$pvalue, method = 'BH')

    #
    # ROUND 2
    #
    # Note: exclude sequences which are very likely to be non-spurious and try again
    #
    cleaned <- filter(spurious, adj_pvalue > 1e-2)
    writeLines(paste('Observations available for sampling:', nrow(cleaned)))
    dfit <- get_sample(cleaned, n_samples, seed)

    writeLines('Round 2 model')
    model2 <- glmmTMB(fixed_model2,
                      ziformula = zi_model2,
                      dispformula = disp_model2,
                      family = distrib_func,
                      control = glmmTMBControl(parallel = 8),
                      data = dfit, verbose = F)

    # some quality control
    fit_summary2 <- summary(model2)
    writeLines('\nSummary for model fit')
    print(fit_summary2)
    writeLines('\nParameter confidence intervals')
    print(confint(model2))
    sink(paste0(output_path, '_R_round2_fit.log'), append = TRUE)
    writeLines('\nSummary for round 2 model fit')
    print(fit_summary2)
    sink()

    if (validate) {
        # simulate residuals plot for model quality inspection
        writeLines('\nSimulating residuals')
        pdf(paste0(output_path, '_R_round2_validation.pdf'))
        layout.matrix <- matrix(c(1, 3, 2, 4), nrow = 2, ncol = 2)
        layout(mat = layout.matrix, heights = c(1, 1), widths = c(1, 1))
        par(cex.lab=0.67)
        simOut2 <- simulateResiduals(model2, plot = F, seed = seed, integerResponse = T, n = 1000)
        writeLines('\nTest uniformity')
        print(testUniformity(simOut2, plot=F))
        plotQQunif(simOut2, testUniformity = T, testOutliers = F, testDispersion = F)
        writeLines('\nTest quantiles')
        print(testQuantiles(simOut2))
        writeLines('\nTest dispersion')
        print(testDispersion(simOut2))
        writeLines('\nTest zero-inflation')
        print(testZeroInflation(simOut2))
        writeLines('\nTest outliers with boostrapping')
        print(testOutliers(simOut2, type = 'bootstrap'))
        dev.off()
    }
    # Recalculate significance for spurious contacts
    spurious['response'] <- predict(model2, spurious, type = 'response')
    spurious['pvalue'] <- 1 - pnbinom(spurious$contacts1m,
                                      size = sigma_func(spurious, model2),
                                      mu = spurious$response)
    spurious['adj_pvalue'] <- p.adjust(spurious$pvalue, method = 'BH')

    writeLines('Calculating significance for all contacts')
    all_contacts <- transform_data(all_contacts)
    all_contacts <- handle_nas(all_contacts, remove_na)

    all_contacts['response'] <- predict(model2, all_contacts, type = 'response')
    all_contacts['pvalue'] <- 1 - pnbinom(all_contacts$contacts1m,
                                          size = sigma_func(all_contacts, model2),
                                          mu = all_contacts$response)
    all_contacts['adj_pvalue'] <- p.adjust(all_contacts$pvalue, method = 'BH')

    list(fitted = spurious, all_contacts = all_contacts,
         fixef = unlist(fixef(model2)), sigma = sigma(model2),
         AIC = AIC(model2), BIC = BIC(model2), logLik = logLik(model2),
         anova = as.matrix(Anova(model2, type = 'III')))
}

find_significant_onepass <- function(spurious, all_contacts, output_path, distrib_func, n_samples, seed,
                                      fixed_model, disp_model, zi_model,
                                      validate=TRUE, remove_na=TRUE) {

    MAX_POINTS <- 1000

    if (missing(distrib_func)) {
        distrib_func <- nbinom2
    }
    if (missing(output_path)) {
        output_path <- "."
    }
    if (!missing(seed)) {
        set.seed(seed)
    }

    # convert strings to formulae
    fixed_model <- as.formula(fixed_model)
    disp_model <- as.formula(disp_model)
    zi_model <- as.formula(zi_model)

    writeLines(paste('Input rows:', nrow(spurious)))

    spurious <- transform_data(spurious)
    spurious <- handle_nas(spurious, remove_na)
    writeLines(paste('Tranformed rows:', nrow(spurious)))

    spurious <- filter(spurious, !intra)
    writeLines(paste('Observations available for sampling:', nrow(spurious)))

    if (missing(n_samples)) {
        n_samples <- nrow(spurious)
    }

    dfit <- get_sample(spurious, n_samples, seed)
    writeLines(paste('Fitting with:', nrow(dfit)))

    # plot points
    if (nrow(dfit) < MAX_POINTS) {
        n_points <- nrow(dfit)
    } else {
        n_points <- MAX_POINTS
    }

    writeLines('Creating correlation plot')
    png(paste0(output_path, '_R_onepass_correlation.png'), width = 1200, height = 800)
    chart.Correlation(sample_n(dfit[, c('contacts1m', 'sites_z', 'length_z', 'cov_z', 'gc_z', 'uf_z')], n_points),
                      histogram = TRUE, pch = 19)
    dev.off()

    #
    # ROUND 1
    #
    # Note: prepare an initial model of our assumed spurious contacts
    #
    model <- glmmTMB(fixed_model,
                     ziformula = zi_model,
                     dispformula = disp_model,
                     family = distrib_func,
                     control = glmmTMBControl(parallel = 8),
                     data = dfit, verbose = F)

    # some quality control
    fit_summary1 <- summary(model)
    writeLines('\nSummary for model fit')
    print(fit_summary1)
    writeLines('\nParameter confidence intervals')
    print(confint(model))
    sink(paste0(output_path, '_R_onepass_fit.log'), append = TRUE)
    writeLines('\nSummary for model fit')
    print(fit_summary1)
    sink()

    if (validate) {
        # simulate residuals plot for model quality inspection
        writeLines('\nSimulating residuals')
        pdf(paste0(output_path, '_R_onepass_validation.pdf'))
        layout.matrix <- matrix(c(1, 3, 2, 4), nrow = 2, ncol = 2)
        layout(mat = layout.matrix, heights = c(1, 1), widths = c(1, 1))
        par(cex.lab=0.67)
        simOut <- simulateResiduals(model, plot = F, seed = seed, integerResponse = T, n = 1000)
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

    # predict expeced contacts (response) from model
    spurious['response'] <- predict(model, spurious, type = 'response')
    # Define p-value as P(X>=expected), where "expected" is that predicted from model
    spurious['pvalue'] <- 1 - pnbinom(spurious$contacts1m,
                                      size = sigma_func(spurious, model),
                                      mu = spurious$response)
    spurious['adj_pvalue'] <- p.adjust(spurious$pvalue, method = 'BH')

    writeLines('Calculating significance for all contacts')
    all_contacts <- transform_data(all_contacts)
    all_contacts <- handle_nas(all_contacts, remove_na)

    all_contacts['response'] <- predict(model, all_contacts, type = 'response')
    all_contacts['pvalue'] <- 1 - pnbinom(all_contacts$contacts1m,
                                          size = sigma_func(all_contacts, model),
                                          mu = all_contacts$response)
    all_contacts['adj_pvalue'] <- p.adjust(all_contacts$pvalue, method = 'BH')

    list(fitted = spurious, all_contacts = all_contacts,
         fixef = unlist(fixef(model)), sigma = sigma(model),
         AIC = AIC(model), BIC = BIC(model), logLik = logLik(model),
         anova = as.matrix(Anova(model, type = 'III')))
}
