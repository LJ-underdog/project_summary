algo-done: 覆盖 §1 3-kernel结构(PRE编译期跳过SiLU/POST条件发射/MAIN必改)、§2 七阶段逐stage HSTU改造(alpha物化+SiLU/softmax双路激活+SiLU显式置零+dS双路+raw_scale→alpha)、§3 fwd副产物契约(SiLU={Q,K,V,dO}无副产物;Softmax加{O,LSE},D由PRE现算)、§4 MAIN双路骨架伪代码+PRE跳过逻辑、§5 复用vs新写清单+6风险点、§6 对pane-2/3接口假设。

关键决策:alpha==FMHA raw_scale槽(两处:STAGE2头+dQ/dK收尾,dV不吃);scale_p折进p与g;SiLU留dsilu因子g(非整张S)以与FMHA寄存器同形;SiLU masked-out必须显式清g(dsilu(0)=0.5),禁用-inf(NaN);fwd LSE为自然对数域,bwd用exp(S-LSE),勿在exp2再乘scale(双scale高危bug)。

未决:R1 SiLU留g的VGPR压力需pane-3真实tile验证;R3 置零强依赖pane-2的block-tile mask谓词;R5 group逐段alpha/scale_p取数归pane-2/3。
