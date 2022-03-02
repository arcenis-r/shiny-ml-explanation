# Create a ggplot2 theme to use throughout the project
ml_eval_theme <- function() {
  theme_bw() + 
    theme(
      plot.title = element_text(hjust = 0.5, size = 18),
      strip.text = element_text(size = 12, color = "white"),
      strip.background = element_rect(fill = "#17468F")
    )
}

plot_chi_sq <- function(df) {
  factor_names <- df %>% select_if(is.factor) %>% names()
  
  chi_sq_dat <- crossing(var1 = factor_names, var2 = factor_names) %>%
    mutate(
      chi_sq_results = map2(
        var1,
        var2,
        ~ select(df, any_of(c(.x, .y))) %>%
          table() %>%
          chisq.test() %>%
          broom::tidy()
      )
    ) %>%
    unnest(chi_sq_results) %>%
    select(var1, var2, p.value) %>%
    pivot_wider(names_from = var2, values_from = p.value) %>%
    column_to_rownames("var1")
  
  chi_sq_dat[!upper.tri(chi_sq_dat)] <- NA

  chi_sq_dat %>%
    rownames_to_column("var1") %>%
    pivot_longer(-var1, names_to = "var2", values_to = "p.value") %>%
    drop_na(p.value) %>%
    ggplot(aes(fct_rev(var2), var1, color = p.value)) +
    geom_point(size = 3) +
    # scale_color_viridis_c(direction = -1) +
    scale_color_gradient(low = "red", high = "gray") +
    labs(
      title = "Chi-square Plot of Categorical Variables",
      color = "P-value"
    ) +
    ml_eval_theme() +
    theme(
      axis.title = element_blank(),
      axis.text.x = element_text(angle = 45, hjust = 1),
      panel.border = element_blank(),
      axis.line = element_line()
    )
}

plot_corr <- function(df) {
  df %>%
    select_if(is.numeric) %>%
    corrr::correlate(method = "spearman", use = "pairwise.complete.obs") %>%
    corrr::rearrange(absolute = FALSE) %>%
    corrr::shave() %>%
    corrr::rplot(colors = c("red", "white", "blue")) +
    labs(
      title = "Correlation Plot of Numeric Variables",
      color = "Correlation"
    ) +
    ml_eval_theme() +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1),
      panel.border = element_blank(),
      axis.line = element_line()
    )
}

plot_class_bal <- function(df, class_col) {
  class_col <- enquo(class_col)
  
  df %>%
    select(!!class_col) %>%
    ggplot(aes(!!class_col, group = !!class_col)) +
    geom_bar(fill = "blue") +
    ml_eval_theme() +
    labs(title = "Group Counts", x = "", y = "")
}
