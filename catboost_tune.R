catboost_fit <- function(data, split, catboost_params, target, cols_to_drop, ...) {
    assert_data_table(data)
    assert_integerish(split, len = data[, .N])
    assert_data_table(catboost_params)
    
    train <- data[split == 0, ]
    val <- data[split == 1, ]
    
    # out feature engineering here
    
    pool_train <- catboost.load_pool(train[, .SD, .SDcols = -cols_to_drop],
                                     label = train[, get(target)])
    pool_val <- catboost.load_pool(val[, .SD, .SDcols = -cols_to_drop],
                                   label = val[, get(target)])                                 
    model <- catboost.train(pool_train, pool_val, 
                            as.list(catboost_params))
    
    preds <- data.table(
        ground_truth = val[, get(target)],
        preds = catboost.predict(model, pool_val)
    )
    
    rmse <- function(obs, pred) sqrt(mean((obs - pred)^2))
    
    data.table(
        nrounds_best = model$tree_count,
        rmse = preds[, rmse(ground_truth, preds)]
    )
}

across_grid <- function(data, split, grid, target, cols_to_drop, ...) {
    metrics <- lapply(seq_len(nrow(grid)),
                      function(i) catboost_fit(data = data, 
                                               split = split, 
                                               target = target, 
                                               cols_to_drop = cols_to_drop,
                                               catboost_params = grid[i, ]))
    metrics <- rbindlist(metrics)
    data.table(grid, metrics)
}

# cv_split should return data.table with columns of 1/0 for each split
# 3-fold example:
# split1 split2 split3
# 0      1      1
# 1      1      0
# 1      0      1
# ....
# 1 = train, 0 = val
splits <- cv_split(train)

catboost_grid <- CJ(
    iterations = 1000,
    learning_rate = 0.05,
    depth = c(8, 9),
    loss_function = "RMSE",
    eval_metric = "RMSE",
    random_seed = 42,
    od_type = 'Iter',
    # metric_period = 50,
    od_wait = 10,
    use_best_model = TRUE,
    logging_level = "Silent"
) 

res <- lapply(splits, 
              function(split) across_grid(train, split, catboost_grid, target, cols_to_drop))
res <- rbindlist(res, idcol = "split")
