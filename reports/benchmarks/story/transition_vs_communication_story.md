   benchmark_family    target            model_name       metric_name  metric_value     direction
   transition_model  AIS->MIA              rna_only sinkhorn_distance     16.297000  lower_better
   transition_model  AIS->MIA                pooled sinkhorn_distance     15.909000  lower_better
   transition_model  AIS->MIA              set_only sinkhorn_distance     15.758000  lower_better
   transition_model  AIS->MIA         graph_of_sets sinkhorn_distance     16.002000  lower_better
communication_relay AIS proxy                pooled        auroc_mean      0.712963 higher_better
communication_relay AIS proxy             graphsage        auroc_mean      0.601852 higher_better
communication_relay AIS proxy             deep_sets        auroc_mean      0.518519 higher_better
communication_relay AIS proxy            focal_only        auroc_mean      0.472222 higher_better
communication_relay AIS proxy  transformer_no_relay        auroc_mean      0.444444 higher_better
communication_relay AIS proxy           stagebridge        auroc_mean      0.425926 higher_better
communication_relay AIS proxy transformer_no_priors        auroc_mean      0.416667 higher_better
communication_relay AIS proxy     graph_transformer        auroc_mean      0.314815 higher_better
