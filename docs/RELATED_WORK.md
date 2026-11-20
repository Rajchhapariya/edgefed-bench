# Related Work

Each entry below was checked against the publisher or arXiv listing in November 2026. The descriptions say only what those listings support.

## Byzantine-robust aggregation

- **Krum and Multi-Krum.** Blanchard, El Mhamdi, Guerraoui and Stainer (NeurIPS 2017) score each update by its distances to its nearest neighbours and keep the lowest-scoring ones. This is our Multi-Krum.
- **Coordinate-wise median and trimmed mean.** Yin, Chen, Kannan and Bartlett (ICML 2018, PMLR 80:5650-5659) analyse both and prove order-optimal statistical error rates for strongly convex losses. These are our two coordinate-wise defences.
- **Heterogeneous data.** Karimireddy, He and Jaggi (ICLR 2022) show that existing robust aggregators can fail to converge when client data differs across workers, even with no attackers. They propose bucketing as a fix. This is why we also measure the honest false-rejection rate and run an alpha sweep.

## Attacks that stay close to honest updates

- **ALIE.** Baruch, Baruch and Goldberg (NeurIPS 2019), "A Little Is Enough", show that small changes inside the natural spread of honest gradients can get past Krum, trimmed mean and Bulyan.
- **IPM.** Xie, Koyejo and Gupta (UAI 2019; PMLR 115:261-270), "Fall of Empires", break coordinate-wise median and Krum by making the inner product between the aggregate and the true gradient negative.

Our sign-flip and Gaussian attacks are easy to detect by distance. ALIE and IPM test whether the INT8 result still holds when attacker updates are hard to tell apart from honest ones.

## Compression of client updates

- **QSGD.** Alistarh, Grubic, Li, Tomioka and Vojnovic (NIPS 2017, pp. 1709-1720) quantize gradients with convergence guarantees and a tunable bits-versus-variance trade-off.
- **signSGD with majority vote.** Bernstein, Zhao, Azizzadenesheli and Anandkumar (ICLR 2019) send only gradient signs and argue that majority vote tolerates up to half the workers being adversarial.
- **Error feedback.** Karimireddy, Rebjock, Stich and Jaggi (ICML 2019) show that scaled sign compression can fail without error feedback, and that error feedback restores SGD's rate. Our 1-bit "Sign" format uses their scaled sign, mean(|x|) times sign(x), without error feedback.

## Compression and Byzantine robustness together

- Ghosh, Maity, Kadhe, Mazumdar and Ramchandran (IEEE Journal on Selected Areas in Information Theory, vol. 2, no. 3, 2021) combine a gradient-norm threshold against Byzantine workers with compressors such as sign and top-k, and analyse error feedback in that setting.
- Zhu and Ling (IEEE Transactions on Signal and Information Processing over Networks, 2023, DOI 10.1109/TSIPN.2023.3265892) study Byzantine-robust distributed learning with compressed communication. An earlier version by the same authors is "BROADCAST" (arXiv:2104.06685, 2021).

**How our work differs.** These papers design new algorithms and prove rates. We do not propose a method. We measure, under a fixed seed protocol, whether plain uniform INT8 (and lower bit widths) changes the accuracy and attacker rejection of three standard robust aggregators. We also log the distances that explain the result.

## Non-IID partitioning

- Hsu, Qi and Brown (arXiv:1909.06335, 2019) build synthetic non-IID client splits with a Dirichlet distribution, with the concentration parameter controlling how similar clients are. Their construction draws a class mix for each client. Ours (`engine/storage.py`) draws, for each class, how that class is shared across clients. Alpha plays the same role in both, but the two are not identical, so alpha values are not directly comparable across papers.

## Earlier references kept from the first version

- McMahan, Moore, Ramage, Hampson and Agüera y Arcas (AISTATS 2017): FedAvg.
- Konečný, McMahan, Yu, Richtárik, Suresh and Bacon (arXiv:1610.05492, 2016): communication efficiency in federated learning.
- Bernstein, Wang, Azizzadenesheli and Anandkumar (ICML 2018): signSGD.
- Raasveldt and Mühleisen (SIGMOD 2019): DuckDB.
