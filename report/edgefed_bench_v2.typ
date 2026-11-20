// EdgeFed-Bench v2 report. Every number comes from metrics.typ and tables.typ,
// which experiments/make_report_assets.py writes from results/results.duckdb.

#import "metrics.typ": m
#import "tables.typ": table_1_robustness, table_2_precision, table_3_wrong_f, table_4_multikrum, table_stats, table_mechanism, table_strong, table_alpha, table_bits, table_cnn, table_cifar, table_long, table_tuning, table_tuned

// ---------------------------------------------------------------- page and type
#set document(title: "INT8 Update Compression and Byzantine-Robust Aggregation in Federated Learning", author: "Raj Chhapariya")
#set page(
  paper: "a4",
  margin: (x: 2.3cm, top: 2.4cm, bottom: 2.4cm),
  numbering: "1",
  number-align: center,
)
#set text(font: "New Computer Modern", size: 10pt, lang: "en")
#set par(justify: true, leading: 0.58em, spacing: 0.9em, first-line-indent: 1.2em)
#set heading(numbering: "1.1")
#show heading.where(level: 1): it => block(above: 1.5em, below: 0.8em)[#set text(size: 11.5pt, weight: "bold"); #it]
#show heading.where(level: 2): it => block(above: 1.2em, below: 0.65em)[#set text(size: 10pt, weight: "bold"); #it]
#set math.equation(numbering: "(1)")
#set list(indent: 1em)
#set enum(indent: 1em)
#show raw.where(block: true): set text(size: 8.5pt)
#show link: set text(fill: rgb("#1f3a93"))
#show ref: set text(fill: rgb("#1f3a93"))

// Figures and tables: captions above tables, below figures, booktabs-style rules.
#set figure(gap: 0.7em)
#show figure.where(kind: table): set figure.caption(position: top)
#show figure.caption: set text(size: 9pt)
#show figure.caption: it => [*#it.supplement #context it.counter.display(it.numbering).* #it.body]
#show figure: set block(above: 1.4em, below: 1.4em)
#set table(align: (x, y) => if x == 0 { left } else { center })
#show table: set text(size: 8.4pt, hyphenate: false)
#show table: set par(justify: false)
#show figure.caption: set par(justify: true, first-line-indent: 0em)
#show table: t => block(width: 100%, stroke: (bottom: 0.8pt))[#t]
#show "+-": sym.plus.minus

// ---------------------------------------------------------------- title block
#align(center)[
  #v(0.4cm)
  #text(size: 16pt, weight: "bold")[INT8 Update Compression and \ Byzantine-Robust Aggregation in Federated Learning]
  #v(0.5cm)
  #text(size: 11pt)[Raj Chhapariya]
  #v(0.1cm)
  #text(size: 9pt)[#link("https://github.com/Rajchhapariya/edgefed-bench")[github.com/Rajchhapariya/edgefed-bench] #h(0.6em) | #h(0.6em) November 2026]
  #v(0.5cm)
]

#pad(x: 1.2cm)[
  #align(center)[#text(size: 10pt, weight: "bold")[Abstract]]
  #v(0.1cm)
  #set text(size: 9pt)
  #set par(first-line-indent: 0em)
  Edge clients in federated learning often quantize their model updates to save uplink bandwidth, while servers use Byzantine-robust aggregators to resist malicious clients. Robust aggregators decide what to keep by comparing updates, so quantization error could blur the line between honest and attacker updates. We measure whether uniform symmetric 8-bit quantization (INT8) weakens Multi-Krum, coordinate-wise median and coordinate-wise trimmed mean. We train on Fashion-MNIST split across 20 non-IID clients and report #m.all_runs runs. INT8 cuts the uplink by #m.savings%. Over #m.grid_seeds paired seeds, the 95% confidence interval of the accuracy difference between INT8 and Float32 lies within #sym.plus.minus#m.stat_margin points in #m.stat_n_equiv_robust of #m.stat_n_robust settings of the three robust aggregators. Coordinate median is the exception: INT8 lowers its accuracy in every setting, by at most #m.stat_max_abs_robust points, and 4-bit quantization lowers it much more. Logged distances explain the main result: the INT8 error of an update is #m.mech_signflip_int8_err_ratio times the distance between two honest updates, so it does not hide sign-flip attackers. Attacks that imitate honest updates (ALIE, IPM) pass Multi-Krum with or without quantization. Rerunning at the best validation hyperparameters keeps the largest robust-aggregator difference at #m.tuned_max_abs_robust points. A CIFAR-10 check with #m.cifar_seeds seeds shows differences up to #m.cifar_max_abs_robust points that are too noisy to test.

  #v(0.15cm)
  *Keywords:* federated learning, Byzantine robustness, quantization, communication efficiency, non-IID data
]

#v(0.3cm)

// ---------------------------------------------------------------- 1
= Introduction

In federated learning, clients train on their own data and send only model updates to a server, which combines them into a global model (McMahan et al., 2017). Edge deployments add two constraints. First, uplink is the scarce direction: phones and IoT devices often have unreliable, slow connections, and the uplink is typically much slower than the downlink (Konečný et al., 2016). Second, plain averaging trusts every client. FedAvg takes the mean of all updates, so one arbitrary update can move the result anywhere (Blanchard et al., 2017).

Quantizing updates to INT8 cuts the upload to about a quarter, but it adds a rounding error to every coordinate. Byzantine-robust aggregators decide what to keep by comparing updates, so that error could in principle make honest and attacker updates harder to tell apart. Our research question is:

#pad(x: 1.2em)[_Does uniform symmetric INT8 quantization of client updates reduce the final accuracy or the attacker rejection of Multi-Krum, coordinate-wise median and coordinate-wise trimmed mean under non-IID Dirichlet label skew ($alpha = 0.5$)?_]

We then check whether the answer holds when attackers imitate honest updates, when label skew is stronger or weaker, when updates use fewer bits, for a second model and a second dataset, over longer training, and at better hyperparameters. Our contributions are:

+ A controlled benchmark of #m.all_runs runs with a fixed seed protocol, stored in one DuckDB file from which every number in this report is generated.
+ Paired statistics over #m.grid_seeds seeds showing no detectable INT8 effect on Multi-Krum and trimmed mean, and a small but consistent loss for coordinate median (@sec-stats).
+ Per-round distance measurements that explain why INT8 does not hide distance-based attackers (@sec-mech).
+ Extended studies of omniscient attacks, label skew, 4-bit and 1-bit updates, a CNN, CIFAR-10, longer training, a hyperparameter check and a rerun at the best setting found (@sec-ext).

We propose no new method. Prior work designs compressed robust algorithms and proves rates (@sec-related); we measure what happens with plain quantization and standard aggregators.

// ---------------------------------------------------------------- 2
= Setup <sec-setup>

== Data and partitioning
We use Fashion-MNIST (Xiao et al., 2017): 60,000 training images, 10,000 test images, 10 classes. The training set is split across $N = 20$ clients with a Dirichlet distribution. The main grid uses $alpha = 0.5$; the label-skew study adds $alpha = 0.1$ and $alpha = 1.0$. Each split is written once to Parquet shards and read back through DuckDB (Raasveldt and Mühleisen, 2019), so every run with the same seed and $alpha$ sees the same data. One extended study uses CIFAR-10 (Krizhevsky, 2009): 50,000 training and 10,000 test colour images of 32#sym.times\32 pixels in 10 classes, split the same way with $alpha = 0.5$.

We measured the skew from the shards (seeds 0 to 4). A client holds on average #m.skew_a0p1_classes classes with at least 1% of its samples at $alpha = 0.1$, #m.skew_a0p5_classes at $alpha = 0.5$ and #m.skew_a1_classes at $alpha = 1.0$. Its largest class makes up on average #m.skew_a0p1_maxshare%, #m.skew_a0p5_maxshare% and #m.skew_a1_maxshare% of its data. For each class, our partitioner draws how that class is shared across clients. Hsu et al. (2019) instead draw a class mix for each client, so $alpha$ values are not directly comparable across papers. For the hyperparameter check only, 10,000 training images are held out as a validation set before partitioning.

== Models and local training
The main model is a multi-layer perceptron with 784 inputs, 128 ReLU hidden units and 10 outputs (101,770 parameters). The second model is a small CNN: two 5#sym.times\5 convolutions with 8 and 16 channels, each followed by ReLU and 2#sym.times\2 max pooling, then fully connected layers of 64 and 10 units (20,522 parameters). For CIFAR-10 the same layout takes 3 input channels (30,138 parameters) and first normalizes each channel with fixed dataset means and standard deviations; without this step it stayed near chance in a short trial. Local training is one epoch of SGD per round with learning rate $eta = 0.05$, momentum 0.9 and batch size 128. The optimizer is created fresh each round, so momentum does not carry across rounds. In round $t$, client $i$ sends
$ Delta bold(w)_i^((t)) = bold(w)_(i, "local")^((t)) - bold(w)_("global")^((t)). $

== Wire formats
In INT8 mode a client maps its update $bold(x) in RR^D$ to signed 8-bit integers with one scale per vector:
$ s = max_j |x_j|, quad S = s / 127, quad q_j = "clip"("round"(x_j \/ S), -127, 127). $ <eq-int8>
The server reconstructs $hat(x)_j = q_j S$, so the error per element is at most $S \/ 2$. One MLP update shrinks from #m.f32_bytes to #m.int8_bytes bytes (one int8 value per parameter plus a 4-byte scale), a #m.savings% cut. INT4 uses the same scheme with integers in $[-7, 7]$, two per byte. The 1-bit Sign format sends $"mean"(|bold(x)|) dot "sign"(bold(x))$, with zero mapped to $+1$ (Bernstein et al., 2018; Karimireddy et al., 2019). Per client and round the payload is #m.bw_kb_float32 KB (Float32), #m.bw_kb_int8 KB (INT8), #m.bw_kb_int4 KB (INT4) and #m.bw_kb_sign KB (Sign). We use no error feedback. Attacker updates are encoded in the same format as honest ones, since the server can only decode that format.

== Attacks
A fixed share of clients, $f \/ N in {10%, 20%, 30%}$, is Byzantine. They use one of four attacks.
- *Sign-flip:* $bold(v)_"byz" = -2.5 bold(v)_"honest"$, which pushes the model against the honest direction.
- *Gaussian:* $bold(v)_"byz" tilde cal(N)(0, sigma^2 bold(I))$ with $sigma = 5$.
- *ALIE* (Baruch et al., 2019): every attacker sends $bold(mu) - z bold(sigma)$, where $bold(mu)$ and $bold(sigma)$ are the coordinate-wise mean and standard deviation of all clients' true updates in that round, $s = floor(N \/ 2 + 1) - f$ and $z = Phi^(-1)((N - s) \/ N)$.
- *IPM* (Xie et al., 2019): every attacker sends $-epsilon bold(mu)$ with $epsilon = 0.1$.
ALIE and IPM attackers see every client's true update. That is the strongest form of these attacks, not the most realistic one.

== Aggregators
*FedAvg* is the unweighted mean of all updates; clients are not weighted by sample count. *Multi-Krum* (Blanchard et al., 2017) scores each update by the sum of squared distances to its $N - f - 2$ nearest neighbours and averages the $N - f$ lowest-scoring updates; this needs $2f + 2 < N$. *Coordinate-wise median* takes the median of each coordinate. *Coordinate-wise trimmed mean* drops the $k = ceil(beta N)$ largest and $k$ smallest values of each coordinate, with $beta = f \/ N$, and averages the rest (Yin et al., 2018). Unless stated otherwise, the server is given the true $f$.

== Experiments and seeds
The main grid has #m.grid_runs runs: 4 aggregators #sym.times 2 attacks (sign-flip, Gaussian) #sym.times 3 attacker shares #sym.times 2 precisions (Float32, INT8) #sym.times #m.grid_seeds seeds (#m.grid_seed_range), plus clean runs without attackers. A wrong-f experiment adds #m.wrong_f_runs runs with the same seeds. Eight extended studies add #m.ext_runs runs and use seeds 0 to 4, except the Fashion-MNIST CNN (#m.cnn_seeds seeds, #m.cnn_rounds rounds) and the CIFAR-10 study (#m.cifar_seeds seeds, #m.cifar_rounds rounds), which are cut to limit CPU time. All other runs last 50 rounds, except the 150-round long-run study.

One seed sets the data split, the model initialization, which clients are Byzantine, the Gaussian noise and the batch order. Runs repeat bit for bit on CPU at a fixed PyTorch thread count, which is stored with each run. Seeds 0 to 4 of the main grid and the wrong-f experiment ran at 10 threads; all other runs ran at 2 threads in 8 parallel processes. A different thread count changes the order of floating-point sums and moves results slightly. Both precisions of a seed always ran at the same thread count, so the paired comparisons are not affected.

== Statistics
We compare INT8 with Float32 for the same aggregator, attack and seed. For each setting we report the mean paired difference, a paired $t$ 95% confidence interval and an exact two-sided sign-flip permutation $p$-value. With $n$ seeds the smallest possible $p$ is $2 \/ 2^n$, which is #m.stat_min_p for $n = #m.stat_n_seeds$. We correct with Holm's method within each aggregator's seven settings; a single family of all #m.stat_n_tests tests could never reach significance at this $n$. We also ask whether the whole confidence interval lies within #sym.plus.minus#m.stat_margin accuracy points. We chose that margin ourselves, after seeing seeds 0 to 4. Seeds 5 to 9 were run afterwards and serve as a confirmation.

// ---------------------------------------------------------------- 3
= Results: Main Grid <sec-main>

At 30% sign-flip with INT8, FedAvg ends at #m.fa30%, which is chance level for 10 classes. Multi-Krum ends at #m.mk30%, trimmed mean at #m.tm30% and coordinate median at #m.cm30% (@tab-robust). At 20% sign-flip, Multi-Krum ends at #m.mk20_f32% with Float32 and #m.mk20% with INT8, a difference smaller than one standard deviation (#m.mk20_sd). @tab-precision compares the two precisions for every aggregator; uplink drops by #m.savings% in each case.

#figure(table_1_robustness, caption: [Final test accuracy (%) with INT8, mean #sym.plus.minus standard deviation over seeds #m.grid_seed_range.]) <tab-robust>

#figure(table_2_precision, caption: [Float32 versus INT8 at 20% sign-flip: final accuracy and uplink per client and round.]) <tab-precision>

#grid(
  columns: (1fr, 1fr),
  gutter: 0.8em,
  align: bottom,
  [#figure(image("figures/fig1_convergence_20pct_signflip.png", width: 100%), caption: [Test accuracy per round at 20% sign-flip (INT8, #m.grid_seeds seeds; band: #sym.plus.minus 1 std).]) <fig-conv>],
  [#figure(image("figures/fig2_accuracy_vs_attacker_share.png", width: 100%), caption: [Final accuracy against attacker share (sign-flip, INT8, #m.grid_seeds seeds).]) <fig-share>],
)

== Wrong assumed number of attackers
A real server does not know how many clients are malicious. We fixed the true number of attackers at $f = 6$ (30% sign-flip, INT8) and told the server $f_"assumed" in {2, 4, 6, 8}$ (@tab-wrongf). With $f_"assumed" = 2$, Multi-Krum falls to #m.mk_wf2% and trimmed mean to #m.tm_wf2%, because attacker updates reach the average. With $f_"assumed" = 8$, they end at #m.mk_wf8% and #m.tm_wf8%, at the cost of discarding some honest updates. In this setup it is safer to set $f_"assumed"$ at or above the expected number of attackers.

#figure(table_3_wrong_f, caption: [Final accuracy (%) when the server assumes the wrong number of attackers. True $f = 6$ (30% sign-flip, INT8), seeds #m.grid_seed_range.]) <tab-wrongf>

== Multi-Krum rejection rates
Under Gaussian noise, Multi-Krum rejected #m.rej_gauss_min% of attacker updates even in its worst run. Sign-flip is harder: averaged over seeds and both precisions, rejection ranges from #m.rej_sf_min% to #m.rej_sf_max% depending on attacker share (@tab-rej). The honest false-rejection rate shows whether label skew makes honest clients look like attackers.

#figure(table_4_multikrum, caption: [Multi-Krum attacker rejection and honest false-rejection rates, averaged over 50 rounds and over seeds.]) <tab-rej>

== Statistical test of the INT8 effect <sec-stats>
For the three robust aggregators, the 95% confidence interval of INT8 minus Float32 lies within #sym.plus.minus#m.stat_margin points in #m.stat_n_equiv_robust of #m.stat_n_robust settings (@tab-stats). The exceptions are: #m.stat_not_equiv_list. Counting only seeds 5 to 9, which we had not seen when we chose the margin, #m.stat_confirm_equiv of #m.stat_confirm_n settings are within it.

One aggregator shows significant differences after Holm correction: #m.stat_sig_aggs. With INT8 it is lower in #m.stat_coordinate_n_neg of #m.stat_coordinate_n settings, and #m.stat_coordinate_n_sig of them are significant; the largest loss is #m.stat_max_abs_robust points. Multi-Krum has #m.stat_multikrum_n_sig significant settings and trimmed mean #m.stat_trimmed_n_sig. With #m.stat_n_seeds seeds, a difference of a few tenths of a point could still go undetected; @tab-stats gives each interval. FedAvg under attack varies so much between seeds that its intervals are wide.

#figure(table_stats, caption: [INT8 minus Float32 accuracy, paired by seed, main grid. $p$: exact permutation test; $p$ (Holm): corrected within each aggregator; last column: whether the 95% CI lies within #sym.plus.minus#m.stat_margin points.]) <tab-stats>

== Why INT8 does not hide attackers <sec-mech>
In every attacked run added in this version, we logged three L2 distances per round: between pairs of honest updates, between honest and attacker updates, and the compression error of each honest update. Seeds 0 to 4 of the main grid and the wrong-f experiment ran before this logging existed and have no such rows.

For Multi-Krum at 20% sign-flip, the INT8 error of an honest update has norm #m.mech_signflip_int8_err. Two honest updates are #m.mech_signflip_int8_pair apart, and an honest and an attacker update #m.mech_signflip_int8_att (@tab-mech, @fig-mech). The error is #m.mech_signflip_int8_err_ratio times the honest-to-honest distance. The ratio of attacker distance to honest distance is #m.mech_signflip_float32_att_ratio with Float32 and #m.mech_signflip_int8_att_ratio with INT8, so the gap Multi-Krum relies on is unchanged. With 1-bit Sign the error grows to #m.mech_signflip_sign_err_ratio times the honest distance, and the attacker ratio is #m.mech_signflip_sign_att_ratio.

ALIE and IPM attackers are closer to honest updates than honest updates are to each other (ratios #m.mech_alie_int8_att_ratio and #m.mech_ipm_int8_att_ratio with INT8), with or without compression. These distances do not explain the coordinate median result above. We did not test why the median is more sensitive.

#figure(table_mechanism, caption: [Mean L2 distances per round, Multi-Krum, 20% attackers, $alpha = 0.5$, MLP. Codec error is the norm of an honest update's compression error.]) <tab-mech>

#figure(image("figures/fig3_mechanism_distances.png", width: 70%), caption: [Distances between updates with INT8 (log scale), Multi-Krum, 20% attackers.]) <fig-mech>

// ---------------------------------------------------------------- 4
= Results: Extended Studies <sec-ext>

== Attacks built to look honest
Multi-Krum rejected #m.sa_alie30_multikrum_rej% of ALIE updates and #m.sa_ipm30_multikrum_rej% of IPM updates at 30%, and @tab-strong shows the same at every share. Under 30% IPM, Multi-Krum ends at #m.sa_ipm30_multikrum_int8% with INT8, below FedAvg at #m.sa_ipm30_fedavg_int8%. Multi-Krum averages $N - f$ updates. Since it kept every attacker update, the updates it dropped were honest, so attackers made up a larger share of its average than of FedAvg's. Coordinate median under 30% IPM ends at #m.sa_ipm30_coordinate_int8%.

With five seeds, no single row can reach $p < 0.05$. Coordinate median has INT8 below Float32 in #m.sa_coordinate_rows_neg of #m.sa_coordinate_rows rows, and in #m.sa_coordinate_n_neg of #m.sa_coordinate_n_pairs seed pairs (#m.sa_coordinate_n_tie equal), the same direction as in the main grid. The other aggregators show no consistent direction: INT8 is lower in #m.sa_multikrum_n_neg of #m.sa_multikrum_n_pairs pairs for Multi-Krum and #m.sa_trimmed_n_neg of #m.sa_trimmed_n_pairs for trimmed mean.

#figure(table_strong, caption: [ALIE and IPM: final accuracy (%), mean #sym.plus.minus std over seeds 0 to 4. Diff is INT8 minus Float32, paired by seed.]) <tab-strong>

== Label skew
In clean runs, stronger skew lowers accuracy for every aggregator. FedAvg with INT8 ends at #m.al_fedavg_none_a0p1% at $alpha = 0.1$, #m.al_fedavg_none_a0p5% at 0.5 and #m.al_fedavg_none_a1% at 1.0. Coordinate median is hit hardest, at #m.al_coordinate_none_a0p1% in clean runs at $alpha = 0.1$ (@tab-alpha).

At $alpha = 0.1$ INT8 also costs coordinate median more. INT8 minus Float32 is #m.al_coordinate_none_a0p1_diff points in clean runs, #m.al_coordinate_signflip_a0p1_diff under sign-flip and #m.al_coordinate_alie_a0p1_diff under ALIE. In each case INT8 was lower on every seed ($p =$ #m.al_coordinate_none_a0p1_p with #m.al_coordinate_none_a0p1_n seeds). Multi-Krum under sign-flip at $alpha = 0.1$ shows #m.al_multikrum_signflip_a0p1_diff with $p =$ #m.al_multikrum_signflip_a0p1_p, which is not evidence of a difference.

#figure(table_alpha, caption: [Final INT8 accuracy (%) and INT8 minus Float32 by Dirichlet $alpha$, seeds 0 to 4.]) <tab-alpha>

== Fewer bits: INT4 and 1-bit sign
For Multi-Krum, Float32 and INT8 end close together, and accuracy falls further with INT4 and again with Sign. At 20% sign-flip Multi-Krum ends at #m.bw_multikrum_signflip_float32% (Float32), #m.bw_multikrum_signflip_int8% (INT8), #m.bw_multikrum_signflip_int4% (INT4) and #m.bw_multikrum_signflip_sign% (Sign) (@tab-bits, @fig-bits).

Coordinate median does not follow this order, and neither does trimmed mean under attack. Trimmed mean at 20% sign-flip ends at #m.bw_trimmed_signflip_int4% with INT4 and #m.bw_trimmed_signflip_sign% with Sign. Coordinate median with INT4 falls to #m.bw_coordinate_none_int4% in clean runs and #m.bw_coordinate_signflip_int4% under sign-flip, while 1-bit Sign keeps #m.bw_coordinate_none_sign% and #m.bw_coordinate_signflip_sign%. A possible cause, which we have not tested, is that 4-bit rounding sets many small coordinates to zero, so the coordinate-wise median of many clients is often zero. None of these formats uses error feedback, which the compression literature uses to recover accuracy (Karimireddy et al., 2019).

#figure(table_bits, caption: [Final accuracy (%) by wire format, mean #sym.plus.minus std over seeds 0 to 4. Payload per client and round in the header.]) <tab-bits>

#figure(image("figures/fig4_bit_width.png", width: 100%), caption: [Final accuracy by wire format and payload size.]) <fig-bits>

== A second model
The CNN, with #m.cnn_seeds seeds and #m.cnn_rounds rounds, shows the same pattern as the MLP (@tab-cnn). At 20% sign-flip Multi-Krum ends at #m.cnn_multikrum_signflip_f32% with Float32 and #m.cnn_multikrum_signflip_int8% with INT8. Among the robust aggregators, coordinate median shows the largest INT8 gap, #m.cnn_coordinate_signflip_f32% with Float32 against #m.cnn_coordinate_signflip_int8% with INT8. With three seeds we report no significance tests for the CNN. On our CPU a CNN round took #m.cnn_cost_ratio times as long as an MLP round at the same thread count.

#figure(table_cnn, caption: [Small CNN: final accuracy (%), mean #sym.plus.minus std over #m.cnn_seeds seeds, #m.cnn_rounds rounds.]) <tab-cnn>

== A second dataset: CIFAR-10
CIFAR-10 is harder for a model this small: after #m.cifar_rounds rounds clean FedAvg reaches #m.cifar_fedavg_none_f32% with Float32 and #m.cifar_fedavg_none_int8% with INT8 (@tab-cifar). FedAvg under 20% sign-flip falls to chance, #m.cifar_fedavg_signflip_f32%. The robust aggregators keep the model training, and Multi-Krum stays closest to the clean result, at #m.cifar_multikrum_signflip_f32% with Float32 and #m.cifar_multikrum_signflip_int8% with INT8. INT8 cuts the payload from #m.cifar_kb_f32 KB to #m.cifar_kb_int8 KB per client and round.

The INT8 differences are larger and noisier than on Fashion-MNIST. For the robust aggregators, INT8 minus Float32 exceeds #m.stat_margin points in #m.cifar_n_over_margin of #m.cifar_n_robust settings and is negative in #m.cifar_n_neg. The largest is #m.cifar_max_abs_robust points (#m.cifar_max_abs_case), while the seed standard deviations of coordinate median and trimmed mean under attack are several points. With #m.cifar_seeds seeds the smallest possible permutation p-value is #m.cifar_min_p, so we cannot test these differences. We read them as a reason to run more seeds on CIFAR-10, not as evidence for or against an INT8 effect. A CIFAR-10 round took #m.cifar_cost_ratio times as long as a Fashion-MNIST CNN round.

#figure(table_cifar, caption: [CIFAR-10, 30,138-parameter CNN: final test accuracy (%), mean #sym.plus.minus std over #m.cifar_seeds seeds (#m.cifar_seed_range), #m.cifar_rounds rounds. Diff is INT8 minus Float32, paired by seed.]) <tab-cifar>

== Longer training
Accuracy keeps rising after round 50: from round 50 to round 150 it gains between #m.long_gain_min and #m.long_gain_max points in every setting (@tab-long). At round 150 under 20% sign-flip, INT8 minus Float32 is #m.long_diff_multikrum points for Multi-Krum, #m.long_diff_trimmed for trimmed mean and #m.long_diff_coordinate for coordinate median. The 50-round results therefore describe models that have not finished training. As a convergence check we also report the gain over the last 50 rounds, from round 100 to 150: #m.long_late_min to #m.long_late_max points. Training is slowing but has not stopped.

#figure(table_long, caption: [Mean test accuracy (%) at rounds 50, 100 and 150, seeds 0 to 4.]) <tab-long>

== Hyperparameter check
On the validation split, our fixed setting (learning rate 0.05, momentum 0.9, batch 128) ranks #m.tune_default_rank of #m.tune_n_configs for clean FedAvg with Float32 (@tab-tune). It is #m.tune_gap points below the best setting, #m.tune_best, at #m.tune_best_acc% (#m.tune_seeds seeds). A batch of 64 means twice as many local steps per round. We did not tune for each aggregator or attack.

#figure(table_tuning, caption: [Validation accuracy (%) of #m.tune_n_configs settings, clean FedAvg, Float32, mean #sym.plus.minus std over #m.tune_seeds seeds.]) <tab-tune>

== Rerun at the best validation setting
We reran clean, 20% sign-flip and 20% ALIE runs for all four aggregators at #m.tuned_setting, with #m.tuned_seeds seeds (@tab-tuned). Test accuracy with Float32 rises by #m.tuned_gain_min to #m.tuned_gain_max points over the batch 128 runs in every setting except FedAvg under sign-flip, which stays unstable across seeds. The comparison between INT8 and Float32 does not change. The largest INT8 minus Float32 difference for a robust aggregator is #m.tuned_max_abs_robust points (#m.tuned_max_abs_case, $p = $ #m.tuned_coordinate_signflip_p). Coordinate median is again lower with INT8 in #m.tuned_coord_n_neg of #m.tuned_coord_n settings, as in the main grid. The conclusions therefore do not depend on the slightly weaker fixed setting.

#figure(table_tuned, caption: [Rerun at the best validation setting, MLP, $alpha = 0.5$, seeds 0 to 4. Batch 128 columns are means of the matching main-grid and ALIE runs. Diff is INT8 minus Float32 at the new setting, paired by seed; $p$ is the exact permutation p-value.]) <tab-tuned>

// ---------------------------------------------------------------- 5
= Related Work <sec-related>

Multi-Krum comes from Blanchard et al. (2017); coordinate-wise median and trimmed mean from Yin et al. (2018). Karimireddy, He and Jaggi (2022) show that robust aggregators can fail under heterogeneous data even without attackers. ALIE (Baruch et al., 2019) and IPM (Xie et al., 2019) are attacks designed to pass such defences. On compression, QSGD (Alistarh et al., 2017) and signSGD with majority vote (Bernstein et al., 2019) reduce communication, and error feedback (Karimireddy et al., 2019) recovers accuracy for biased compressors. Ghosh et al. (2021) and Zhu and Ling (2023) combine compression with Byzantine robustness and prove rates for new algorithms. We propose no method. We measure whether plain uniform quantization changes three standard robust aggregators under a fixed seed protocol, and we log the distances that explain the result.

// ---------------------------------------------------------------- 6
= Limitations

+ *One machine.* All clients are simulated on one host (Intel Core i7-13620H), not on phones or embedded boards. We do not model network jitter, packet loss or thermal throttling.
+ *Two small datasets, small models.* The main grid uses Fashion-MNIST. The CNN studies have fewer seeds than the MLP, and the CIFAR-10 study is too small to test its INT8 differences. Larger networks or other datasets may behave differently.
+ *Seeds.* The main grid and the wrong-f experiment use #m.grid_seeds seeds; the extended studies use 5, the Fashion-MNIST CNN #m.cnn_seeds and the CIFAR-10 study #m.cifar_seeds. The equivalence margin was chosen after seeing seeds 0 to 4.
+ *The server knows $f$.* Outside the wrong-f experiment, the aggregator is given the true number of attackers, which is the best case for the defences.
+ *Omniscient strong attacks.* ALIE and IPM attackers see every client's true update.
+ *Hyperparameters.* They were fixed before the study and checked only for clean FedAvg. The rerun at the best setting covers 3 scenarios and 2 wire formats, not the full grid.
+ *No error feedback, no differential privacy.* INT4 and Sign run without error feedback. We add no clipping or privacy noise, which could interact with the rounding error.
+ *50 rounds.* Accuracy is still rising at round 50, and still rising slowly at round 150.

// ---------------------------------------------------------------- 7
= Conclusion

INT8 quantization of client updates cut the uplink by #m.savings% and did not change the accuracy of Multi-Krum or trimmed mean beyond what #m.grid_seeds paired seeds can detect. Coordinate median lost a small but consistent amount, more under strong label skew and much more with INT4. The INT8 error is small next to the distances Multi-Krum relies on, which explains why distance-based attackers stay visible. Attacks that imitate honest updates pass Multi-Krum regardless of quantization, so the choice of wire format matters less than the choice of defence against them. The same pattern held at the best validation hyperparameters. On CIFAR-10 the INT8 differences were larger, but three seeds are too few to tell them from noise; that is the first experiment we would extend.

// ---------------------------------------------------------------- references
#v(0.4cm)
#heading(numbering: none)[References]
#set par(first-line-indent: 0em, hanging-indent: 1.2em, spacing: 0.6em)
#set text(size: 9pt)

Alistarh, D., Grubic, D., Li, J., Tomioka, R., & Vojnovic, M. (2017). QSGD: Communication-efficient SGD via gradient quantization and encoding. _NIPS 2017_, 1709--1720.

Baruch, M., Baruch, G., & Goldberg, Y. (2019). A little is enough: Circumventing defenses for distributed learning. _NeurIPS 2019_.

Bernstein, J., Wang, Y.-X., Azizzadenesheli, K., & Anandkumar, A. (2018). signSGD: Compressed optimisation for non-convex problems. _ICML 2018_, 560--569.

Bernstein, J., Zhao, J., Azizzadenesheli, K., & Anandkumar, A. (2019). signSGD with majority vote is communication efficient and fault tolerant. _ICLR 2019_.

Blanchard, P., El Mhamdi, E. M., Guerraoui, R., & Stainer, J. (2017). Machine learning with adversaries: Byzantine tolerant gradient descent. _NeurIPS 2017_.

Ghosh, A., Maity, R. K., Kadhe, S., Mazumdar, A., & Ramchandran, K. (2021). Communication-efficient and Byzantine-robust distributed learning with error feedback. _IEEE Journal on Selected Areas in Information Theory_, 2(3).

Hsu, T.-M. H., Qi, H., & Brown, M. (2019). Measuring the effects of non-identical data distribution for federated visual classification. _arXiv:1909.06335_.

Karimireddy, S. P., He, L., & Jaggi, M. (2022). Byzantine-robust learning on heterogeneous datasets via bucketing. _ICLR 2022_.

Karimireddy, S. P., Rebjock, Q., Stich, S. U., & Jaggi, M. (2019). Error feedback fixes SignSGD and other gradient compression schemes. _ICML 2019_.

Konečný, J., McMahan, H. B., Yu, F. X., Richtárik, P., Suresh, A. T., & Bacon, D. (2016). Federated learning: Strategies for improving communication efficiency. _arXiv:1610.05492_.

Krizhevsky, A. (2009). Learning multiple layers of features from tiny images. Technical report, University of Toronto.

McMahan, B., Moore, E., Ramage, D., Hampson, S., & Agüera y Arcas, B. (2017). Communication-efficient learning of deep networks from decentralized data. _AISTATS 2017_, 1273--1282.

Raasveldt, M., & Mühleisen, H. (2019). DuckDB: An embeddable analytical database. _SIGMOD 2019_, 1981--1984.

Xiao, H., Rasul, K., & Vollgraf, R. (2017). Fashion-MNIST: A novel image dataset for benchmarking machine learning algorithms. _arXiv:1708.07747_.

Xie, C., Koyejo, O., & Gupta, I. (2019). Fall of empires: Breaking Byzantine-tolerant SGD by inner product manipulation. _UAI 2019_, PMLR 115:261--270.

Yin, D., Chen, Y., Kannan, R., & Bartlett, P. (2018). Byzantine-robust distributed learning: Towards optimal statistical rates. _ICML 2018_, PMLR 80:5650--5659.

Zhu, H., & Ling, Q. (2023). Byzantine-robust distributed learning with compression. _IEEE Transactions on Signal and Information Processing over Networks_. DOI 10.1109/TSIPN.2023.3265892.

// ---------------------------------------------------------------- appendix
#pagebreak()
#set par(first-line-indent: 0em, hanging-indent: 0em)
#set text(size: 10pt)
#counter(heading).update(0)
#set heading(numbering: "A.1")
= Reproduction

Every table, figure and number in this report is generated from `results/results.duckdb` by `experiments/make_report_assets.py`. Code and data: #link("https://github.com/Rajchhapariya/edgefed-bench")[github.com/Rajchhapariya/edgefed-bench].

```bash
pip install -r requirements.txt
python -m unittest discover tests -v
python experiments/run_grid.py --experiment everything --threads 10
python experiments/partition_stats.py
python experiments/make_report_assets.py
typst compile report/edgefed_bench_v2.typ report/edgefed_bench_v2.pdf
```

The runner resumes where it stopped and can be split across processes with `--shard k/K`; see the README, Section 7.3. Hardware and software versions are in `results/ENVIRONMENT.md`.
