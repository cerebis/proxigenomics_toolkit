# suppress console warnings about function name clashes
suppressMessages(library(readr))
suppressMessages(library(dplyr))
suppressMessages(library(sjmisc))
suppressMessages(library(tibble))
suppressMessages(library(glmmTMB))
suppressMessages(library(car))
suppressMessages(library(PerformanceAnalytics))
suppressMessages(library(DHARMa))

transform_data = function(data) {

  # drop rows with zeros
  d = tibble(subset(data, sites_u>0 & cov_u>0 & sites_v>0 & cov_v>0))

  # make sure the columns are full precision to minimise overflows
  d = mutate(d, 
      across(
        c(length_u, 
          length_v, 
          sites_u, 
          sites_v, 
          cov_u, 
          cov_v, 
          gc_u, 
          gc_v), 
        as.double))

  # Prepare data, where sequence (u) and genome_bin (v) variables are taken as products
  d = select(
      sjmisc::std(
        mutate(d,
           # endogenous variable will use a ZIF model rather than one-flated
               contacts1m = contacts - 1,
               # log transformation of exogenous
               length = log(length_u * length_v),
               sites = log(sites_u * sites_v),
               cov = log(cov_u * cov_v),
               gc = log(gc_u * gc_v)),
        # standardise exogenous
        length, 
        sites, 
        cov, 
        gc, 
        robust='sd'),
      # drop unstandardised exog columns
      -c('length', 'sites', 'cov', 'gc'))

  # return
  d
}


find_significant = function(fitted, all_contacts, output_path, distrib_func, n_samples, seed,
                            fixed_model, disp_model, zif_model) {

  MAX_POINTS = 1000

  if (missing(distrib_func)) {
    distrib_func = nbinom2
  }
  if (missing(output_path)) {
    output_path = "."
  }
  if (!missing(seed)) {
    set.seed(seed)
  }

  # convert strings to formulae
  fixed_model = as.formula(fixed_model)
  disp_model = as.formula(disp_model)
  zif_model = as.formula(zif_model)

  writeLines(paste('Input rows:', nrow(fitted)))
  
  fitted = transform_data(fitted)
  writeLines(paste('Tranformed rows:', nrow(fitted)))

  dfit = subset(fitted, intra==FALSE)
  writeLines(paste('Observations available for sampling:', nrow(dfit)))

  if (!missing(n_samples)) {
    dfit = sample_n(dfit, n_samples)
  }
  writeLines(paste('Fitting with:', nrow(dfit)))

  if (nrow(dfit) < MAX_POINTS) {
    n_points = nrow(dfit)
  } else {
    n_points = MAX_POINTS
  }

  writeLines('Creating correlation plot')
  png(paste0(output_path, '_R_correlation.png'), width=1200, height=800)
  chart.Correlation(sample_n(dfit[, c('contacts1m', 'sites_z', 'length_z', 'cov_z', 'gc_z')], n_points),
                    histogram=TRUE, pch=19)
  dev.off()

  writeLines('Fitting model')
  modfit = glmmTMB(fixed_model,
                   ziformula = zif_model,
                   dispformula = disp_model,
                   family = distrib_func,
                   data=dfit, verbose=F)

  # some quality control
  fit_summary1 = summary(modfit)
  writeLines('\nSummary for model fit')
  print(fit_summary1)
  writeLines('\nParameter confidence intervals')
  print(confint(modfit))
  sink(paste0(output_path, '_R_fit.log'), append=TRUE)
  writeLines('\nSummary for model fit')
  print(fit_summary1)
  sink()

  # simulate residuals plot for model quality inspection
  png(paste0(output_path, '_R_simulated-residuals.png'), width=1200, height=800)
  par(mfrow=c(1,2))
  simulateResiduals(modfit, plot=T, seed=seed)
  dev.off()

  # predict contact response from model and assign p/q values of observed contacts
  fitted['response'] = predict(modfit, fitted, type='response')
  fitted['pvalue'] = pnbinom(fitted$contacts1m, size=sigma(modfit), mu=fitted$response)
  fitted['qvalue'] = 1 - fitted$pvalue
  
  writeLines('Calculating significance for all contacts')
  all_contacts = transform_data(all_contacts)
  all_contacts['response'] = predict(modfit, all_contacts, type='response')
  all_contacts['pvalue'] = pnbinom(all_contacts$contacts1m, size=sigma(modfit), mu=all_contacts$response)
  all_contacts['qvalue'] = 1 - all_contacts$pvalue

  list(fitted=fitted, all_contacts=all_contacts, 
    fixef=unlist(fixef(modfit)), sigma=sigma(modfit), 
    AIC=AIC(modfit), BIC=BIC(modfit), logLik=logLik(modfit), 
    anova=as.matrix(Anova(modfit, type='III')))
}
