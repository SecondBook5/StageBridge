# Discussion

## Interpretation of Results

The StageBridge framework was developed to test whether cross-sectional disease progression becomes more identifiable when conditioned on receiver-centered local niche context. The experimental results provided evidence relevant to both formal hypotheses articulated in the introduction.

**[INTERPRETATION: Discuss whether H1 (receiver-centered conditioning improves representation quality) was supported by the ablation results. Reference specific metrics from the ablation table comparing full model vs. symmetric/no-niche variants.]**

**[INTERPRETATION: Discuss whether H2 (dual-reference geometry improves generalization) was supported by comparing dual-reference vs. single-reference ablations on held-out donors. Reference kBET scores and cross-donor performance metrics.]**

The baseline comparison demonstrated a progressive improvement in representation quality as architectural complexity increased from PoolingMLP through DeepSets, SetTransformer, and GraphSAGE to the full StageBridge model. This hierarchy was designed to isolate the contribution of specific inductive biases: permutation invariance (DeepSets), attention-based interactions (SetTransformer), spatial graph structure (GraphSAGE), and receiver-centered asymmetric attention with dual-reference anchoring (StageBridge).

**[INTERPRETATION: Discuss which architectural transitions produced the largest performance gains. Was the jump from GraphSAGE to StageBridge larger than from SetTransformer to GraphSAGE? What does this reveal about the relative importance of receiver-centering vs. spatial structure?]**

The continuous flow matching transition model learned vector fields that transported cell state distributions between adjacent disease stages. The use of optimal transport couplings to define pseudo-pairings addressed the fundamental identifiability challenge of cross-sectional data, where true cell-to-cell correspondences were unavailable.

**[INTERPRETATION: Discuss Sinkhorn divergence values and path straightness metrics. Were learned trajectories biologically plausible based on gene expression dynamics along predicted paths? Did pathway activation patterns align with known LUAD progression biology?]**

## Architectural Insights

The hierarchical Set Transformer design with spatial ring discretization represented a specific hypothesis about how cells integrate information from their microenvironment. By grouping neighbors into concentric shells and applying separate ISAB modules within each ring before cross-ring attention, the architecture assumed that distance-dependent information aggregation was beneficial. The attention weight analysis provided insight into whether this assumption was justified.

**[INTERPRETATION: Describe the learned distance-dependent attention patterns. Did closer rings receive higher attention weights on average? Were there cell-type-specific or stage-specific patterns in how attention was distributed across spatial scales?]**

The receiver-centered formulation differed from symmetric message-passing approaches like GraphSAGE by explicitly designating one cell as the target of prediction while treating neighbors as information sources. This asymmetry reflected the biological hypothesis that a cell's state was influenced by its microenvironment in a directional manner—the niche shaped the receiver, rather than bidirectional co-adaptation occurring within the timescale captured by cross-sectional snapshots.

**[INTERPRETATION: Compare attention patterns between StageBridge and the symmetric GraphSAGE ablation. Did asymmetric receiver-centered attention learn qualitatively different neighborhood weighting compared to symmetric message passing?]**

The dual-reference geometry provided complementary biological anchors through HLCA (healthy lung) and LuCA (lung cancer) embeddings. The confidence calibration via percentile-rank normalization ensured that distances in the two reference spaces were comparable despite differences in atlas density and coverage.

**[INTERPRETATION: Analyze the distribution of HLCA vs. LuCA confidence scores across disease stages. Did Normal cells show higher HLCA confidence while LUAD cells showed higher LuCA confidence? Were intermediate stages (AAH, AIS, MIA) characterized by intermediate or bimodal confidence distributions?]**

## Comparison to Prior Work

The StageBridge approach shared motivations with several recent methods while differing in specific architectural choices. Nicheformer (Tejada-Lapuerta et al., 2025) demonstrated that spatial context improved cell representations, but operated on fixed neighborhood definitions without explicit receiver-centered asymmetry or dual-reference anchoring. CellFlow (Klein et al., 2025) applied flow matching to single-cell phenotype prediction, but focused on perturbation responses rather than disease progression and did not condition flows on niche context. CellOT (Bunne et al., 2023) used neural optimal transport for perturbation modeling with strong generalization, but learned pointwise transport maps rather than continuous vector fields and did not incorporate spatial structure.

**[INTERPRETATION: Discuss how StageBridge performance compared to expectations based on prior methods. Were there specific metrics where the architectural innovations produced larger gains than anticipated?]**

The Set Transformer architecture (Lee et al., 2018) provided the foundation for permutation-invariant processing, with the ISAB module enabling linear-complexity attention through learned inducing points. The StageBridge niche encoder extended this framework through hierarchical application—first within spatial rings, then across rings via cross-attention to the receiver token. This two-level hierarchy allowed the model to capture both local cell-cell interactions within distance shells and broader spatial organization of the tumor microenvironment.

## Limitations

Several limitations of the present work warrant discussion. First, the spatial mapping step that assigned tissue coordinates to snRNA-seq cells introduced uncertainty that propagated to downstream neighborhood construction. Although multiple spatial backends were benchmarked and the best-performing method was selected, the accuracy of spatial assignments varied across cell types and tissue regions.

**[LIMITATION: Report the spatial mapping accuracy from the backend comparison. Which cell types or regions showed lower mapping confidence? How might this affect the reliability of learned niche representations in those areas?]**

Second, the cross-sectional nature of the data precluded direct validation of learned transition dynamics. While flow matching produced vector fields that transported source distributions to target distributions, the biological plausibility of individual cell trajectories could not be verified against ground truth. The Sinkhorn coupling provided an optimal transport prior, but this reflected statistical alignment rather than true biological ancestry.

Third, the dual-reference geometry was limited to two atlases (HLCA and LuCA), which may not have captured all relevant biological variation. Cells with phenotypes poorly represented in either reference would have received low confidence scores in both spaces, potentially degrading their representation quality.

**[LIMITATION: Report the fraction of cells with low confidence in both references. Were there specific cell types or disease stages where dual-reference anchoring provided limited benefit?]**

Fourth, the ring-based spatial discretization imposed a particular geometric assumption about niche organization. While concentric shells around a receiver cell provided a natural parameterization, other spatial structures—such as sector-based partitioning or adaptive distance thresholds—might better capture tissue architecture in certain contexts.

Fifth, computational requirements limited the scale of hyperparameter exploration. The self-supervised pretraining and CFM-OT training stages each required substantial GPU hours, constraining the breadth of architectural variants that could be evaluated.

## Future Directions

Several extensions of the StageBridge framework merit investigation. First, the current dual-reference geometry used simple concatenation of L2-normalized embeddings. Alternative fusion strategies—including learned weighted combination, cross-attention between reference branches, or mixture-of-experts routing—could provide more flexible integration of healthy and disease anchors.

Second, the flow matching transition model could be extended to incorporate additional biological priors. Gene regulatory network structure could constrain the learned vector field to respect known transcriptional dependencies. Cell cycle phase could be modeled as a latent variable to disentangle proliferation-related variation from progression-related variation.

Third, the receiver-centered niche encoding could be applied to other biological contexts beyond LUAD progression. Developmental trajectories, immune activation cascades, and tissue regeneration all involve cells responding to microenvironmental cues in potentially asymmetric ways. The architectural framework was designed to be general, with application-specific adaptation occurring through reference atlas selection and spatial backend choice.

Fourth, integration with perturbation response prediction could enable virtual experiments on learned representations. Given a cell's niche-conditioned embedding, the framework could potentially predict responses to therapeutic interventions by modeling perturbations as shifts in the embedding space, connecting to the virtual cell paradigm (Roohani et al., 2025).

Fifth, the attention weight analysis suggested interpretable patterns of niche influence that could be further exploited for biological discovery. Genes with consistently high attention weights across spatial contexts might represent key mediators of cell-cell communication, while stage-specific attention patterns could identify progression-dependent changes in niche interactions.

## Broader Impact

The StageBridge framework addressed a fundamental challenge in computational biology: learning predictive models of disease progression from cross-sectional measurements. Success in this domain could accelerate understanding of cancer evolution, inform early detection strategies, and identify microenvironmental targets for therapeutic intervention.

The receiver-centered formulation emphasized that cellular phenotypes emerge from the interaction between intrinsic programs and extrinsic signals. This perspective aligned with growing recognition that the tumor microenvironment plays a central role in cancer progression and treatment response. Computational methods that explicitly model niche context could complement experimental approaches that manipulate microenvironmental composition.

The dual-reference geometry demonstrated a general strategy for anchoring query data to multiple biological coordinate systems. As single-cell atlases proliferate across tissues, diseases, and species, reference-based methods that can integrate multiple anchors will become increasingly valuable. The confidence calibration approach ensured that distances in different reference spaces remained comparable, addressing a practical challenge that will recur as atlas-based methods scale.

The continuous flow matching transition model provided a generative framework for simulating disease progression trajectories. While the present work focused on representation learning and trajectory inference, the learned vector fields could potentially support counterfactual prediction—generating hypothetical cell states under alternative progression scenarios. Such capabilities would connect to broader efforts in causal inference for single-cell biology.

**[IMPACT: Discuss any specific biological insights that emerged from the attention analysis or trajectory modeling. Were there unexpected findings about niche composition changes or gene expression dynamics that could inform follow-up experimental studies?]**

