# suppress console warnings about function name clashes
suppressMessages(library(readr))
suppressMessages(library(dplyr))
suppressMessages(library(sjmisc))
suppressMessages(library(tibble))
suppressMessages(library(glmmTMB))
suppressMessages(library(car))
suppressMessages(library(PerformanceAnalytics))
suppressMessages(library(DHARMa))
# requires KernSmooth library

transform_data <- function(data) {

    # drop rows with zeros
    d <- tibble(subset(data, sites_u > 0 & sites_v > 0 &
                             cov_u > 0 & cov_v > 0 &
                             uf_u > 0 & uf_v > 0))

    # make sure the columns are full precision to minimise overflows
    d <- mutate(d,
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
    d <- select(
         sjmisc::std(
             mutate(d,
                    # endogenous variable will use a ZIF model rather than one-flated
                    contacts1m = contacts - 1,
                    # log transformation of exogenous
                    length = log(length_u * length_v),
                    sites = log(sites_u * sites_v),
                    cov = log(cov_u * cov_v),
                    # arcsine transformation for proportional variables
                    uf = asin(uf_u * uf_v),
                    gc = gc_v - gc_u),
             # standardise exogenous
             length,
             sites,
             cov,
             gc,
             uf,
             robust = 'sd'),
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

find_significant <- function(spurious, all_contacts, output_path, distrib_func, n_samples, seed,
                            fixed_model, disp_model, zi_model, validate=TRUE) {

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
    writeLines(paste('Tranformed rows:', nrow(spurious)))

    dfit <- subset(spurious, intra == FALSE)
    writeLines(paste('Observations available for sampling:', nrow(dfit)))

    if (!missing(n_samples)) {
        dfit <- sample_n(dfit, n_samples)
        writeLines(paste('Reduced observation set size'))
    }
    writeLines(paste('Fitting with:', nrow(dfit)))

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

    writeLines('Fitting model')
    modfit <- glmmTMB(fixed_model,
                      ziformula = zi_model,
                      dispformula = disp_model,
                      family = distrib_func,
                      control = glmmTMBControl(parallel = 8),
                      data = dfit, verbose = F)

    # some quality control
    fit_summary1 <- summary(modfit)
    writeLines('\nSummary for model fit')
    print(fit_summary1)
    writeLines('\nParameter confidence intervals')
    print(confint(modfit))
    sink(paste0(output_path, '_R_fit.log'), append = TRUE)
    writeLines('\nSummary for model fit')
    print(fit_summary1)
    sink()

    if (validate) {
        # simulate residuals plot for model quality inspection
        writeLines('\nSimulating residuals')
        pdf(paste0(output_path, '_R_validation.pdf'))
        layout.matrix <- matrix(c(1, 3, 2, 4), nrow = 2, ncol = 2)
        layout(mat = layout.matrix, heights = c(1, 1), widths = c(1, 1))
        par(cex.lab=0.67)
        simOut <- simulateResiduals(modfit, plot = F, seed = seed, integerResponse = T, n = 1000)
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
    spurious['response'] <- predict(modfit, spurious, type = 'response')
    # Define p-value as P(X>=expected), where "expected" is that predicted from model
    spurious['pvalue'] <- 1 - pnbinom(spurious$contacts1m,
                                      size = sigma_func(spurious, modfit), mu = spurious$response)

    writeLines('Calculating significance for all contacts')
    #write_csv(all_contacts, paste0(output_path, 'temp.csv'))
    all_contacts <- transform_data(all_contacts)
    all_contacts['response'] <- predict(modfit, all_contacts, type = 'response')
    all_contacts['pvalue'] <- 1 - pnbinom(all_contacts$contacts1m,
                                          size = sigma_func(all_contacts, modfit), mu = all_contacts$response)

    list(fitted = spurious, all_contacts = all_contacts,
         fixef = unlist(fixef(modfit)), sigma = sigma(modfit),
         AIC = AIC(modfit), BIC = BIC(modfit), logLik = logLik(modfit),
         anova = as.matrix(Anova(modfit, type = 'III')))
}
